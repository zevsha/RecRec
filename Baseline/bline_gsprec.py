import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import polars as pl



def detect_schema(path, probe_lines=20):
    """Peek at the first few lines to figure out field names (legacy vs
    2023-format Amazon review JSONL)."""
    user_candidates = ["user_id", "reviewerID"]
    item_candidates = ["parent_asin", "asin"]
    time_candidates = ["timestamp", "unixReviewTime"]
    with open(path, "r") as f:
        for _ in range(probe_lines):
            line = f.readline()
            if not line:
                break
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            u = next((c for c in user_candidates if c in r), None)
            i = next((c for c in item_candidates if c in r), None)
            t = next((c for c in time_candidates if c in r), None)
            if u and i and t:
                return u, i, t
    raise ValueError(f"Could not detect user/item/timestamp fields in first "
                      f"{probe_lines} lines of {path}")


def fast_load_and_filter(path, min_inter=5, max_users=25000, item_min_inter=1):
    """Vectorized load + filter, matching the spec:
      1) keep users with >= min_inter interactions (user-side only)
      2) cap to max_users by most recent activity (max timestamp per user)
      3) item pool = every item touched by the kept users (no re-shrinking
         unless item_min_inter > 1, applied once, not recursively)

    Returns: triples_idx (list of (u_idx, i_idx, ts)), u2idx, i2idx, seqs
    """
    user_col, item_col, time_col = detect_schema(path)
    print(f"[data] detected fields: user={user_col!r} item={item_col!r} time={time_col!r}")

    lf = (
        pl.scan_ndjson(path, ignore_errors=True)
        .select([
            pl.col(user_col).cast(pl.Utf8, strict=False).alias("user"),
            pl.col(item_col).cast(pl.Utf8, strict=False).alias("item"),
            pl.col(time_col).cast(pl.Int64, strict=False).alias("ts"),
        ])
        .drop_nulls()
    )

    # Dedup (user,item), keep earliest timestamp — single vectorized groupby.
    df = lf.group_by(["user", "item"]).agg(pl.col("ts").min()).collect(streaming=True)
    print(f"[data] raw unique (user,item) pairs: {df.height:,}")

    # --- Step 1: user-side min_inter filter only ---
    user_counts = df.group_by("user").agg(pl.len().alias("cnt"))
    keep_users = user_counts.filter(pl.col("cnt") >= min_inter)["user"]
    df = df.filter(pl.col("user").is_in(keep_users))
    print(f"[data] users with >= {min_inter} interactions: {keep_users.len():,}")

    # --- Step 2: cap to max_users by most recent activity ---
    if max_users is not None:
        user_last = df.group_by("user").agg(pl.col("ts").max().alias("last_ts"))
        if user_last.height > max_users:
            top_users = user_last.sort("last_ts", descending=True).head(max_users)["user"]
            df = df.filter(pl.col("user").is_in(top_users))
    n_users = df["user"].n_unique()
    print(f"[data] users after cap: {n_users:,}")

    # --- Step 3: item pool = items touched by kept users. Optional single-pass
    # item-side filter (does NOT recurse back onto users). ---
    if item_min_inter and item_min_inter > 1:
        item_counts = df.group_by("item").agg(pl.len().alias("cnt"))
        keep_items = item_counts.filter(pl.col("cnt") >= item_min_inter)["item"]
        df = df.filter(pl.col("item").is_in(keep_items))

    n_items = df["item"].n_unique()
    print(f"[data] final: users={n_users:,} items={n_items:,} interactions={df.height:,} "
          f"density={df.height/(n_users*n_items):.4%}")

    # --- Build contiguous integer ids via join (vectorized, no python dict scans) ---
    users = df["user"].unique().sort()
    items = df["item"].unique().sort()
    users_df = pl.DataFrame({"user": users, "u_idx": np.arange(len(users), dtype=np.int64)})
    items_df = pl.DataFrame({"item": items, "i_idx": np.arange(len(items), dtype=np.int64)})
    df = df.join(users_df, on="user").join(items_df, on="item")

    u2idx = dict(zip(users_df["user"].to_list(), users_df["u_idx"].to_list()))
    i2idx = dict(zip(items_df["item"].to_list(), items_df["i_idx"].to_list()))

    # --- Per-user chronologically-ordered item sequences (vectorized sort + groupby) ---
    seq_df = df.sort(["u_idx", "ts"])
    grouped = seq_df.group_by("u_idx", maintain_order=True).agg(pl.col("i_idx"))
    seqs = {u: [] for u in range(len(users))}
    for u, items_list in zip(grouped["u_idx"].to_list(), grouped["i_idx"].to_list()):
        seqs[u] = items_list

    triples_idx = list(zip(
        df["u_idx"].to_list(), df["i_idx"].to_list(), df["ts"].to_list()
    ))

    return triples_idx, u2idx, i2idx, seqs


def split_per_user(seqs, rng, train_ratio=0.8, val_ratio=0.1):
    """8:1:1 random split of each user's interactions. Users with < 3 items
    contribute only to train."""
    train, val, test = {}, {}, {}
    for u, items in seqs.items():
        items = list(items)
        if len(items) < 3:
            train[u] = set(items)
            val[u] = set()
            test[u] = set()
            continue
        idx = rng.permutation(len(items))
        n_test = max(1, int(round(len(items) * (1 - train_ratio - val_ratio))))
        n_val = max(1, int(round(len(items) * val_ratio)))
        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]
        train[u] = {items[k] for k in train_idx}
        val[u] = {items[k] for k in val_idx}
        test[u] = {items[k] for k in test_idx}
    return train, val, test


# ---------------------------------------------------------------------------
# Graph construction (Sec 3.2)
# ---------------------------------------------------------------------------
def build_X(train, m, n):
    """Binary user-item matrix from train interactions."""
    rows, cols = [], []
    for u, its in train.items():
        for i in its:
            rows.append(u)
            cols.append(i)
    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(m, n))


def build_S_prime(seqs, train_by_user, n):
    """S' = symmetrize(S), where s_ij=1 if i directly precedes j in ANY
    user's sequence. We restrict to TRAIN interactions to avoid leakage.
    """
    rows, cols = [], []
    for u, items in seqs.items():
        allowed = train_by_user.get(u, set())
        # Filter sequence to train items only, preserving order.
        filt = [i for i in items if i in allowed]
        for a, b in zip(filt[:-1], filt[1:]):
            if a == b:
                continue
            rows.append(a); cols.append(b)
    if not rows:
        return sp.csr_matrix((n, n), dtype=np.float32)
    data = np.ones(len(rows), dtype=np.float32)
    S = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    S.data[:] = 1.0  # binarize
    S_prime = S + S.T
    S_prime.data[:] = 1.0
    S_prime.setdiag(0)
    S_prime.eliminate_zeros()
    return S_prime.tocsr()


def multi_hop_diffusion(S_prime, d=2, alpha=0.4):
    """S^(d) = sum_{k=1..d} alpha^(k-1) (S')^k  (Eq. 2)."""
    S_d = S_prime.copy().astype(np.float32)
    power = S_prime.copy().astype(np.float32)
    for k in range(2, d + 1):
        power = (power @ S_prime).astype(np.float32)
        S_d = S_d + (alpha ** (k - 1)) * power
    return S_d.tocsr()


def sym_normalize(M):
    """D^(-1/2) M D^(-1/2) for symmetric non-negative M."""
    deg = np.asarray(M.sum(axis=1)).flatten()
    d_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    d_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return (D_inv_sqrt @ M @ D_inv_sqrt).tocsr()


def build_unified_A(X, S_tilde):
    """A = [[0, X], [X^T, S~]]  (Eq. 5)."""
    m, n = X.shape
    zero_uu = sp.csr_matrix((m, m), dtype=np.float32)
    top = sp.hstack([zero_uu, X], format="csr")
    bot = sp.hstack([X.T, S_tilde], format="csr")
    return sp.vstack([top, bot], format="csr")


# ---------------------------------------------------------------------------
# Spectral filtering (Sec 3.3)
# ---------------------------------------------------------------------------
def truncated_eig(A_norm, r):
    """Return (eigvals of L, eigvecs) for the r smallest eigenpairs of L=I-A_norm.
    Trick: smallest eigenvalues of L == largest of A_norm.
    """
    r = min(r, A_norm.shape[0] - 2)
    # eigsh needs symmetric input; A_norm is symmetric by construction.
    vals_P, vecs = eigsh(A_norm, k=r, which="LA")
    # Sort ascending in L-space.
    vals_L = 1.0 - vals_P
    order = np.argsort(vals_L)
    return vals_L[order], vecs[:, order]


def bandpass_response(vals_L, c, w):
    lam_min, lam_max = vals_L.min(), vals_L.max()
    if lam_max - lam_min < 1e-12:
        lam_bar = np.zeros_like(vals_L)
    else:
        lam_bar = (vals_L - lam_min) / (lam_max - lam_min)
    return np.exp(-((lam_bar - c) ** 2) / w)


def compute_scores(X, U, vals_L, m, n, c, w, phi):
    """Compute Y = phi * F_BP + (1-phi) * F_LP  (Eqs. 8-11)."""
    U_I = U[m:, :]  # item block of eigenvectors, shape (n, r)

    # Interaction normalization for spectral branches (Eq. 4).
    deg_I = np.asarray(X.sum(axis=0)).flatten()
    d_i_inv_sqrt = np.zeros_like(deg_I)
    nz = deg_I > 0
    d_i_inv_sqrt[nz] = 1.0 / np.sqrt(deg_I[nz])
    D_I_inv_sqrt = sp.diags(d_i_inv_sqrt)

    # --- Bandpass  F_BP = X D_I^{-1/2} U_I G_BP U_I^T D_I^{-1/2}  (Eq. 8)
    g_bp = bandpass_response(vals_L, c, w)
    # Compute right-to-left: keep intermediate matrices dense but small (n x r).
    XDI = (X @ D_I_inv_sqrt).toarray().astype(np.float32)  # (m, n)
    UI_g = U_I * g_bp[np.newaxis, :]                       # (n, r)
    tmp = XDI @ UI_g                                        # (m, r)
    tmp = tmp @ U_I.T                                       # (m, n)
    F_BP = tmp * d_i_inv_sqrt[np.newaxis, :]                # (m, n)

    # --- Lowpass  F_LP over augmented signal [C_U, X]  (Eq. 9-10)
    # Normalize X to X_tilde_U for building C_U (Eq. 4 left).
    deg_U = np.asarray(X.sum(axis=1)).flatten()
    d_u_inv_sqrt = np.zeros_like(deg_U)
    nzu = deg_U > 0
    d_u_inv_sqrt[nzu] = 1.0 / np.sqrt(deg_U[nzu])
    X_tilde_U = sp.diags(d_u_inv_sqrt) @ X                  # (m, n)
    C_U = (X_tilde_U @ X_tilde_U.T).toarray().astype(np.float32)  # (m, m) dense

    X_b = np.concatenate([C_U, X.toarray().astype(np.float32)], axis=1)  # (m, m+n)
    col_sums = X_b.sum(axis=0)
    d_b_inv_sqrt = np.zeros_like(col_sums)
    nzb = col_sums > 0
    d_b_inv_sqrt[nzb] = 1.0 / np.sqrt(col_sums[nzb])
    d_b_sqrt = np.sqrt(col_sums)

    Xb_scaled = X_b * d_b_inv_sqrt[np.newaxis, :]           # (m, m+n)
    tmp = Xb_scaled @ U                                     # (m, r)
    tmp = tmp @ U.T                                         # (m, m+n)
    F_LP_full = tmp * d_b_sqrt[np.newaxis, :]               # (m, m+n)
    F_LP = F_LP_full[:, m:]                                 # (m, n)

    Y = phi * F_BP + (1.0 - phi) * F_LP
    return Y


# ---------------------------------------------------------------------------
# Evaluation (full ranking, exclude train items)
# ---------------------------------------------------------------------------
def evaluate(scores, train, target, ks=(1,5,10)):
    """Compute NDCG@k and MRR@k averaged over users with non-empty target."""
    m, n = scores.shape
    max_k = max(ks)
    ndcg = {k: [] for k in ks}
    mrr = {k: [] for k in ks}
    precision = {k: [] for k in ks}
    hr = {k: [] for k in ks}

    for u in range(m):
        tgt = target.get(u, set())
        if not tgt:
            continue
        row = scores[u].copy()
        # Mask training items.
        tr = train.get(u, set())
        if tr:
            row[list(tr)] = -np.inf
        # Top-max_k
        if max_k < n:
            idx = np.argpartition(-row, max_k)[:max_k]
            idx = idx[np.argsort(-row[idx])]
        else:
            idx = np.argsort(-row)
        hits = np.array([1.0 if idx[j] in tgt else 0.0 for j in range(max_k)])

        for k in ks:
            hits_k = hits[:k]
            # NDCG (binary relevance)
            hit_count=hits_k.sum()
            if hits_k.sum() == 0:
                ndcg[k].append(0.0)
                mrr[k].append(0.0)
                hr[k].append(0.0)
                precision[k].append(0.0)
                continue
            gains = hits_k / np.log2(np.arange(2, k + 2))
            dcg = gains.sum()
            n_rel = min(len(tgt), k)
            idcg = (1.0 / np.log2(np.arange(2, n_rel + 2))).sum()
            ndcg[k].append(dcg / idcg)
            # MRR
            first = np.argmax(hits_k > 0) if hits_k.sum() > 0 else -1
            mrr[k].append(1.0 / (first + 1) if hits_k[first] > 0 else 0.0)
            # HR@k: 1 if at least one relevant item in top-k, else 0
            hr[k].append(1.0 if hit_count > 0 else 0.0)

            # Precision@k: fraction of top-k that are relevant
            precision[k].append(hit_count / k)

    out = {}
    for k in ks:
        out[f"NDCG@{k}"] = float(np.mean(ndcg[k])) if ndcg[k] else 0.0
        out[f"MRR@{k}"]  = float(np.mean(mrr[k]))  if mrr[k]  else 0.0
        out[f"HR@{k}"] = float(np.mean(hr[k])) if hr[k] else 0.0
        out[f"Prec@{k}"]  = float(np.mean(precision[k]))  if precision[k]  else 0.0
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", default='/storage/Pervez/GRFM/new_impl/Datasets/Beauty_and_Personal_Care.jsonl', help="Path to Amazon reviews JSONL")
    ap.add_argument("--max_users", type=int, default=25000)
    ap.add_argument("--min_inter", type=int, default=5,
                     help="min interactions per user (user-side filter only)")
    ap.add_argument("--item_min_inter", type=int, default=2,
                     help="optional single-pass min interactions per item "
                          "(1 = no item filtering, item pool = all items "
                          "touched by kept users)")
    ap.add_argument("--seed", type=int, default=42)
    # GSPRec hyperparameters (paper Beauty defaults)
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--r", type=int, default=512)
    ap.add_argument("--c", type=float, default=0.8)
    ap.add_argument("--w", type=float, default=0.3)
    ap.add_argument("--phi", type=float, default=0.5)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    print(f"[data] loading {args.reviews}")
    triples, u2idx, i2idx, seqs = fast_load_and_filter(
        args.reviews,
        min_inter=args.min_inter,
        max_users=args.max_users,
        item_min_inter=args.item_min_inter,
    )
    m, n = len(u2idx), len(i2idx)
    print(f"[data] users={m}  items={n}  interactions={len(triples)}  "
          f"density={len(triples)/(m*n):.4%}  ({time.time()-t0:.1f}s)")

    print("[split] 8:1:1 per-user random")
    train, val, test = split_per_user(seqs, rng)

    print("[graph] building X, S', S^(d), S~, A, L")
    X = build_X(train, m, n)
    S_prime = build_S_prime(seqs, train, n)
    print(f"        S' nnz={S_prime.nnz}")
    S_d = multi_hop_diffusion(S_prime, d=args.d, alpha=args.alpha)
    S_tilde = sym_normalize(S_d)
    A = build_unified_A(X, S_tilde)
    A_norm = sym_normalize(A)  # this is D^{-1/2} A D^{-1/2}, i.e. I - L

    print(f"[eig] computing {args.r} smallest eigenpairs of L (via largest of A_norm)")
    t1 = time.time()
    vals_L, U = truncated_eig(A_norm, args.r)
    print(f"      done in {time.time()-t1:.1f}s  |  lam range=[{vals_L.min():.4f}, {vals_L.max():.4f}]")

    print(f"[filter] c={args.c} w={args.w} phi={args.phi}")
    scores = compute_scores(X, U, vals_L, m, n,
                            c=args.c, w=args.w, phi=args.phi)

    print("[eval] validation (for tuning)")
    val_metrics = evaluate(scores, train, val)
    for k, v in val_metrics.items():
        print(f"       {k}: {v:.4f}")

    print("[eval] test (exclude train items only)")
    test_metrics = evaluate(scores, train, test)
    for k, v in test_metrics.items():
        print(f"       {k}: {v:.4f}")

    print(f"[done] total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
