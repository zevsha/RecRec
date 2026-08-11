import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



def detect_schema(path, probe_lines=20):
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
    raise ValueError(f"Could not detect schema in {path}")


def fast_load_and_filter(path, min_inter=5, max_users=None):
    """User-side min-5 only, cap by most recent activity, item pool =
    every item those users touched."""
    user_col, item_col, time_col = detect_schema(path)
    print(f"[data] fields: user={user_col!r} item={item_col!r} time={time_col!r}")

    lf = (
        pl.scan_ndjson(path, ignore_errors=True)
        .select([
            pl.col(user_col).cast(pl.Utf8, strict=False).alias("user"),
            pl.col(item_col).cast(pl.Utf8, strict=False).alias("item"),
            pl.col(time_col).cast(pl.Int64, strict=False).alias("ts"),
        ])
        .drop_nulls()
    )
    df = lf.group_by(["user", "item"]).agg(pl.col("ts").min()).collect(streaming=True)
    print(f"[data] raw unique (user,item) pairs: {df.height:,}")

    user_counts = df.group_by("user").agg(pl.len().alias("cnt"))
    keep_users = user_counts.filter(pl.col("cnt") >= min_inter)["user"]
    df = df.filter(pl.col("user").is_in(keep_users))
    print(f"[data] users with >= {min_inter} interactions: {keep_users.len():,}")

    if max_users is not None:
        user_last = df.group_by("user").agg(pl.col("ts").max().alias("last_ts"))
        if user_last.height > max_users:
            top_users = user_last.sort("last_ts", descending=True).head(max_users)["user"]
            df = df.filter(pl.col("user").is_in(top_users))

    n_users = df["user"].n_unique()
    n_items = df["item"].n_unique()
    print(f"[data] final: users={n_users:,} items={n_items:,} "
          f"interactions={df.height:,} density={df.height/(n_users*n_items):.4%}")

    
    users = df["user"].unique().sort()
    items = df["item"].unique().sort()
    users_df = pl.DataFrame({"user": users,
                             "u_idx": np.arange(len(users), dtype=np.int64)})
    items_df = pl.DataFrame({"item": items,
                             "i_idx": np.arange(1, len(items) + 1, dtype=np.int64)})
    df = df.join(users_df, on="user").join(items_df, on="item")

    # Per-user chronologically ordered item sequence.
    seq_df = df.sort(["u_idx", "ts"])
    grouped = seq_df.group_by("u_idx", maintain_order=True).agg(pl.col("i_idx"))
    seqs = {u: itms for u, itms in
            zip(grouped["u_idx"].to_list(), grouped["i_idx"].to_list())}
    return seqs, len(users), len(items)


def prepare_loo(seqs, min_seq_len=4):
    """For each user with >= min_seq_len items:
        full_seq = [i0, ..., i_{n-1}]
        test_target = i_{n-1}
        val_target  = i_{n-2}
        train_seq   = [i0, ..., i_{n-3}]   (used for shifted CE training)
        val_input   = train_seq
        test_input  = train_seq + [val_target]
    """
    train_seqs, val_data, test_data = {}, {}, {}
    skipped = 0
    for u, full in seqs.items():
        if len(full) < min_seq_len:
            skipped += 1
            continue
        test_target = full[-1]
        val_target = full[-2]
        train_seq = full[:-2]
        train_seqs[u] = train_seq
        val_data[u] = {"input": train_seq,
                        "target": val_target,
                        "exclude": set(train_seq)}
        test_data[u] = {"input": train_seq + [val_target],
                         "target": test_target,
                         "exclude": set(train_seq)}
    print(f"[loo] {len(train_seqs):,} users kept  ({skipped:,} skipped with <{min_seq_len} items)")
    return train_seqs, val_data, test_data


class SeqDataset(Dataset):
    """Shifted next-item pairs for SASRec-style training. Input at position t
    is train_seq[t], target is train_seq[t+1]. Left-padded to max_len."""
    def __init__(self, train_seqs, max_len, pad_idx=0):
        self.users = list(train_seqs.keys())
        self.seqs = [train_seqs[u] for u in self.users]
        self.max_len = max_len
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.users)

    def __getitem__(self, i):
        seq = self.seqs[i]
        # Take the last (max_len + 1) items so we have max_len (input, target) pairs
        seq = seq[-(self.max_len + 1):]
        input_seq = seq[:-1]
        target_seq = seq[1:]
        pad = self.max_len - len(input_seq)
        input_seq = [self.pad_idx] * pad + input_seq
        target_seq = [self.pad_idx] * pad + target_seq
        mask = [0.0] * pad + [1.0] * (self.max_len - pad)
        return (torch.tensor(input_seq, dtype=torch.long),
                torch.tensor(target_seq, dtype=torch.long),
                torch.tensor(mask, dtype=torch.float32))


class MultiHeadAttention(nn.Module):
    def __init__(self, d, H, dropout):
        super().__init__()
        assert d % H == 0
        self.d = d
        self.H = H
        self.d_p = d // H
        self.W_Q = nn.Parameter(torch.randn(H, d, self.d_p) * 0.02)
        self.W_K = nn.Parameter(torch.randn(H, d, self.d_p) * 0.02)
        self.W_V = nn.Parameter(torch.randn(H, d, self.d_p) * 0.02)
        self.W_O = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, causal_mask, key_pad_mask):
        # x: (B, T, d);  causal_mask: (T, T) bool;  key_pad_mask: (B, T) bool
        B, T, _ = x.shape
        Q = torch.einsum('btd,hde->bhte', x, self.W_Q)
        K = torch.einsum('btd,hde->bhte', x, self.W_K)
        V = torch.einsum('btd,hde->bhte', x, self.W_V)
        scores = torch.einsum('bhti,bhji->bhtj', Q, K) / math.sqrt(self.d_p)
        scores = scores.masked_fill(causal_mask[None, None], float('-inf'))
        scores = scores.masked_fill(key_pad_mask[:, None, None, :], float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # all-masked rows -> 0
        attn = self.drop(attn)
        out = torch.einsum('bhtj,bhjd->bhtd', attn, V)
        out = out.transpose(1, 2).reshape(B, T, self.d)  # concat heads
        return self.W_O(out)


class TransformerBlock(nn.Module):
    """Standard SASRec block (post-LN)."""
    def __init__(self, d, H, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d, H, dropout)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ffn1 = nn.Linear(d, d)
        self.ffn2 = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, causal, key_pad):
        a = self.attn(x, causal, key_pad)
        x = self.ln1(x + self.drop(a))
        f = self.ffn2(F.relu(self.ffn1(x)))
        x = self.ln2(x + self.drop(f))
        return x


class SASRecBackbone(nn.Module):
    def __init__(self, num_items, d, H, num_blocks, max_len, dropout, pad_idx=0):
        super().__init__()
        self.d, self.H, self.num_blocks = d, H, num_blocks
        self.max_len, self.pad_idx = max_len, pad_idx
        self.item_emb = nn.Embedding(num_items + 1, d, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, d)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(d, H, dropout)
                                     for _ in range(num_blocks)])
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[pad_idx].zero_()

    def _embed(self, seq):
        B, T = seq.shape
        positions = torch.arange(T, device=seq.device).unsqueeze(0).expand(B, T)
        x = self.item_emb(seq) + self.pos_emb(positions)
        return self.emb_drop(x)

    def _masks(self, seq):
        T = seq.size(1)
        causal = torch.triu(torch.ones(T, T, device=seq.device, dtype=torch.bool),
                            diagonal=1)
        key_pad = (seq == self.pad_idx)
        return causal, key_pad

    def forward(self, seq, run_blocks=None):
        """run_blocks: iterable of block indices to run. Default: all."""
        x = self._embed(seq)
        causal, key_pad = self._masks(seq)
        blocks = self.blocks if run_blocks is None else [self.blocks[i] for i in run_blocks]
        for blk in blocks:
            x = blk(x, causal, key_pad)
        return x


class SASRecModel(nn.Module):
    """Standard SASRec: predict next item at every position via dot-product with
    the item embedding table."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, seq, positions=None):
        """positions: None -> all T positions (training).
        int (e.g. -1) -> only that position, returns (B, V+1) — used at eval
        to avoid materializing the full (B, T, V) logits tensor."""
        x = self.backbone(seq)                                  # (B, T, d)
        if positions is not None:
            x = x[:, positions, :]                               # (B, d)
            return x @ self.backbone.item_emb.weight.T          # (B, V+1)
        logits = x @ self.backbone.item_emb.weight.T            # (B, T, V+1)
        return logits

    def score_items(self, seq, item_ids):
        """Score only a small set of candidate items per position — never
        materializes the full (B, T, V) tensor. item_ids: (B, T, K).
        Returns (B, T, K)."""
        x = self.backbone(seq)                                   # (B, T, d)
        emb = self.backbone.item_emb(item_ids)                   # (B, T, K, d)
        return (x.unsqueeze(2) * emb).sum(-1)                    # (B, T, K)


class FAMEBlock(nn.Module):
    def __init__(self, d, H, N, dropout):
        super().__init__()
        assert d % H == 0
        self.d, self.H, self.N = d, H, N
        self.d_p = d // H
        self.W_Q = nn.Parameter(torch.randn(H, N, d, self.d_p) * 0.02)      # experts
        self.W_K = nn.Parameter(torch.randn(H, d, self.d_p) * 0.02)         # retained from SASRec
        self.W_V = nn.Parameter(torch.randn(H, d, self.d_p) * 0.02)         # retained from SASRec
        self.W_exp = nn.Parameter(torch.randn(H, N * self.d_p, N) * 0.02)   # router per head
        # FFN' shared across heads (paper Sec 4.2.3)
        self.ffn1 = nn.Linear(self.d_p, self.d_p)
        self.ffn2 = nn.Linear(self.d_p, self.d_p)
        self.ln = nn.LayerNorm(self.d_p)
        self.drop = nn.Dropout(dropout)

    def load_kv_from_pretrained(self, mha: MultiHeadAttention):
        with torch.no_grad():
            self.W_K.copy_(mha.W_K.data)
            self.W_V.copy_(mha.W_V.data)

    def forward(self, x, causal_mask, key_pad_mask):
        # x: (B, T, d) -> per-head output F_head: (B, H, T, d')
        B, T, _ = x.shape
        H, N, d_p = self.H, self.N, self.d_p

        # Per-expert Q  (B, T, H, N, d')
        Q = torch.einsum('btd,hnde->bthne', x, self.W_Q)
        # Per-head K, V  (B, H, T, d')
        K = torch.einsum('btd,hde->bhte', x, self.W_K)
        V = torch.einsum('btd,hde->bhte', x, self.W_V)

        # Attention scores per head per expert  (B, H, N, T, T)
        Q_t = Q.permute(0, 2, 3, 1, 4)                                    # (B, H, N, T, d')
        scores = torch.einsum('bhnti,bhji->bhntj', Q_t, K) / math.sqrt(d_p)
        scores = scores.masked_fill(causal_mask[None, None, None], float('-inf'))
        scores = scores.masked_fill(key_pad_mask[:, None, None, None, :], float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.drop(attn)

        # Per-expert item repr  (B, H, N, T, d')
        f_expert = torch.einsum('bhntj,bhjd->bhntd', attn, V)
        # Router: concat over N experts, project to N via W_exp per head
        f_expert_t = f_expert.permute(0, 1, 3, 2, 4).contiguous()         # (B, H, T, N, d')
        f_cat = f_expert_t.view(B, H, T, N * d_p)                         # (B, H, T, N*d')
        beta_logits = torch.einsum('bhtd,hdn->bhtn', f_cat, self.W_exp)   # (B, H, T, N)
        beta = F.softmax(beta_logits, dim=-1)
        # Integrated head repr (Eq. 17)
        f_head = torch.einsum('bhtn,bhtnd->bhtd', beta, f_expert_t)       # (B, H, T, d')

        # Per-head shared FFN' with residual + LN (Eq. 7)
        ffn_out = self.ffn2(F.relu(self.ffn1(f_head)))
        F_head = self.ln(f_head + self.drop(ffn_out))                     # (B, H, T, d')
        return F_head


class FAMEModel(nn.Module):
    """SASRec backbone (blocks 0..L-2) + FAMEBlock as final block + per-head
    prediction + head-gate (Eq. 10-11). Per-position gate is used at every
    timestep — natural generalization of Eq. 10 since W_g is shared."""
    def __init__(self, backbone: SASRecBackbone, N, dropout):
        super().__init__()
        self.backbone = backbone
        d, H = backbone.d, backbone.H
        self.d, self.H, self.N = d, H, N
        self.d_p = d // H
        self.fame_block = FAMEBlock(d, H, N, dropout)
        # Retain K, V from pretrained SASRec's last block (Sec 4.4.2).
        self.fame_block.load_kv_from_pretrained(backbone.blocks[-1].attn)
        # Per-head item sub-embedding projection W_f^(h) (Eq. 9)
        self.W_f = nn.Parameter(torch.randn(H, d, self.d_p) * 0.02)
        # Head gate (Eq. 10)
        self.W_g = nn.Parameter(torch.randn(d, H) * 0.02)
        self.b_g = nn.Parameter(torch.zeros(H))

    def forward(self, seq, positions=None):
        """positions: None -> all T positions (training, needed for shifted CE).
        int (e.g. -1) -> only that position, returns (B, V+1) — used at eval
        to avoid ever materializing a (B, H, T, V) tensor.

        Regardless of `positions`, this loops over heads when projecting to
        the vocabulary instead of computing all H head-logit tensors at once
        (which is what blows up memory: (B, H, T, V) vs looping H times over
        (B, T, V) and immediately reducing). This makes training memory
        roughly O(B*T*V) instead of O(B*H*T*V)."""
        B, T = seq.shape
        # Run all backbone blocks EXCEPT the last (which FAMEBlock replaces).
        L = self.backbone.num_blocks
        x = self.backbone(seq, run_blocks=range(L - 1))                    # (B, T, d)
        causal, key_pad = self.backbone._masks(seq)
        F_head = self.fame_block(x, causal, key_pad)                       # (B, H, T, d')

        if positions is not None:
            F_head = F_head[:, :, positions, :]                            # (B, H, d')
            F_head = F_head.unsqueeze(2)                                   # (B, H, 1, d')
            T_eff = 1
        else:
            T_eff = T

        # Per-position gate: concat heads -> (B, T_eff, d), project to H
        F_cat = F_head.permute(0, 2, 1, 3).contiguous().view(B, T_eff, self.d)
        g = F.softmax(F_cat @ self.W_g + self.b_g, dim=-1)                 # (B, T_eff, H)

        item_w = self.backbone.item_emb.weight                             # (V+1, d)
        # Loop over heads: never hold more than one head's (B, T_eff, V) at once.
        logits = None
        for h in range(self.H):
            x_h = item_w @ self.W_f[h]                                     # (V+1, d')
            score_h = torch.einsum('btd,vd->btv', F_head[:, h], x_h)       # (B, T_eff, V+1)
            contrib = g[:, :, h:h + 1] * score_h
            logits = contrib if logits is None else logits + contrib
        if positions is not None:
            logits = logits.squeeze(1)                                    # (B, V+1)
        return logits

    def _shared_hidden(self, seq):
        """Run backbone + FAMEBlock, return (F_head (B,H,T,d'), g (B,T,H))."""
        B, T = seq.shape
        L = self.backbone.num_blocks
        x = self.backbone(seq, run_blocks=range(L - 1))                    # (B, T, d)
        causal, key_pad = self.backbone._masks(seq)
        F_head = self.fame_block(x, causal, key_pad)                       # (B, H, T, d')
        F_cat = F_head.permute(0, 2, 1, 3).contiguous().view(B, T, self.d)
        g = F.softmax(F_cat @ self.W_g + self.b_g, dim=-1)                 # (B, T, H)
        return F_head, g

    def score_items(self, seq, item_ids):
        """Score only a small set of candidate items per position — never
        materializes the full (B, T, V) tensor. item_ids: (B, T, K).
        Returns (B, T, K)."""
        F_head, g = self._shared_hidden(seq)                               # (B,H,T,d'), (B,T,H)
        item_w = self.backbone.item_emb(item_ids)                          # (B, T, K, d)
        scores = None
        for h in range(self.H):
            item_h = item_w @ self.W_f[h]                                  # (B, T, K, d')
            score_h = (F_head[:, h].unsqueeze(2) * item_h).sum(-1)         # (B, T, K)
            contrib = g[:, :, h:h + 1] * score_h
            scores = contrib if scores is None else scores + contrib
        return scores


def train_epoch(model, loader, optim, device, pad_idx=0,
                grad_accum_steps=1, use_amp=False, scaler=None,
                num_items=None, num_neg=0):
    """num_neg=0 -> full-softmax over the whole vocabulary (fine for small/
    medium catalogs, e.g. Beauty's ~12k items).
    num_neg>0   -> sampled softmax: score only the true target + num_neg
    random negatives (shared across the batch, resampled every step). Cost
    becomes independent of vocab size — needed once V gets into the 100k+
    range (e.g. Electronics has 117,853 items, where full softmax OOMs even
    at small batch sizes because backward must retain a (B,T,V) tensor)."""
    model.train()
    total, n = 0.0, 0
    optim.zero_grad()
    for step, (inp, tgt, msk) in enumerate(loader):
        inp, tgt, msk = inp.to(device), tgt.to(device), msk.to(device)
        B, T = tgt.shape
        with torch.autocast(device_type=device.type, enabled=use_amp,
                            dtype=torch.bfloat16 if device.type == "cpu" else torch.float16):
            if num_neg and num_neg > 0:
                neg_ids = torch.randint(1, num_items + 1, (num_neg,), device=device)
                candidates = torch.cat([
                    tgt.unsqueeze(-1),                                      # (B,T,1) true target at index 0
                    neg_ids.view(1, 1, -1).expand(B, T, -1),                # (B,T,num_neg)
                ], dim=-1)
                scores = model.score_items(inp, candidates)                # (B, T, 1+num_neg)
                labels = torch.zeros(B * T, dtype=torch.long, device=device)
                loss_raw = F.cross_entropy(scores.reshape(-1, scores.size(-1)),
                                           labels, reduction='none')
            else:
                logits = model(inp)                                        # (B, T, V+1)
                V = logits.size(-1)
                loss_raw = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1),
                                           reduction='none')
            loss = (loss_raw * msk.reshape(-1)).sum() / msk.sum().clamp(min=1)
            loss = loss / grad_accum_steps
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (step + 1) % grad_accum_steps == 0:
            if use_amp and scaler is not None:
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()
            optim.zero_grad()
        total += loss.item() * grad_accum_steps * inp.size(0)
        n += inp.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, data, max_len, device, pad_idx=0, batch=256, ks=(1, 5, 10)):
    """Full-ranking LOO evaluation. `data`: dict u -> {input, target, exclude}."""
    model.eval()
    hr = {k: 0.0 for k in ks}
    precision = {k: 0.0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    users = list(data.keys())
    for start in range(0, len(users), batch):
        chunk = users[start:start + batch]
        inputs, excludes, targets = [], [], []
        for u in chunk:
            seq = data[u]["input"][-max_len:]
            pad = max_len - len(seq)
            inputs.append([pad_idx] * pad + list(seq))
            excludes.append(data[u]["exclude"])
            targets.append(data[u]["target"])
        seq_t = torch.tensor(inputs, dtype=torch.long, device=device)
        logits = model(seq_t, positions=-1)                                # (B, V+1)
        # Mask excluded items + padding index
        for i, exc in enumerate(excludes):
            if exc:
                logits[i, list(exc)] = float('-inf')
        logits[:, pad_idx] = float('-inf')
        tgt_t = torch.tensor(targets, dtype=torch.long, device=device)
        tgt_scores = logits.gather(1, tgt_t.unsqueeze(1)).squeeze(1)       # (B,)
        # Rank = 1 + #items with strictly greater score
        rank = (logits > tgt_scores.unsqueeze(1)).sum(dim=1) + 1
        rank_np = rank.cpu().numpy()
        for k in ks:
            hit = (rank_np <= k)
            hr[k] += hit.sum()
            ndcg[k] += (hit / np.log2(rank_np + 1)).sum()
            precision[k]+=(hit / k).sum()
    n = len(users)
    return ({f"HR@{k}": hr[k] / n for k in ks} |
            {f"NDCG@{k}": ndcg[k] / n for k in ks} |
            {f"Prec@{k}": precision[k] / n for k in ks})


def fmt(metrics):
    return "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())


def run_stage(name, model, train_loader, val_data, test_data,
              epochs, patience, lr, max_len, device, log_every=1,
              grad_accum_steps=1, use_amp=False, eval_batch=256,
              num_items=None, num_neg=0):
    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    best_val, best_test, best_epoch, stale = -1, None, -1, 0
    best_state = None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, train_loader, optim, device,
                           grad_accum_steps=grad_accum_steps, use_amp=use_amp,
                           scaler=scaler, num_items=num_items, num_neg=num_neg)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        val = evaluate(model, val_data, max_len, device, batch=eval_batch)
        key = "NDCG@10"
        improved = val[key] > best_val
        msg = (f"[{name}] ep {ep:3d}  loss {loss:.4f}  "
               f"val {fmt(val)}  ({time.time()-t0:.1f}s)")
        if improved:
            best_val = val[key]
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            msg += "  *"
        else:
            stale += 1
        if ep % log_every == 0 or improved:
            print(msg)
        if stale >= patience:
            print(f"[{name}] early stopping at epoch {ep} (best val NDCG@10 = {best_val:.4f} at ep {best_epoch})")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[{name}] evaluating best checkpoint on test set...")
    test = evaluate(model, test_data, max_len, device, batch=eval_batch)
    print(f"[{name}] TEST  {fmt(test)}")
    return test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", default='/storage/Pervez/GRFM/new_impl/Datasets/Beauty_and_Personal_Care.jsonl')
    ap.add_argument("--max_users", type=int, default=None)
    ap.add_argument("--min_inter", type=int, default=5)
    # Model
    ap.add_argument("--d", type=int, default=64, help="embedding dim")
    ap.add_argument("--H", type=int, default=2, help="num heads")
    ap.add_argument("--N", type=int, default=4, help="num experts per head")
    ap.add_argument("--num_blocks", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=50)
    ap.add_argument("--dropout", type=float, default=0.5)
    # Training
    ap.add_argument("--pretrain_epochs", type=int, default=200)
    ap.add_argument("--finetune_epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--eval_batch", type=int, default=512,
                     help="user-batch size for full-ranking eval (independent of train batch)")
    ap.add_argument("--grad_accum_steps", type=int, default=1,
                     help="accumulate gradients over N steps to keep effective "
                          "batch size while lowering per-step memory")
    ap.add_argument("--amp", action="store_true",
                     help="mixed precision training (roughly halves activation memory on GPU)")
    ap.add_argument("--num_neg", type=int, default=100,
                     help="sampled-softmax negatives per position, shared across the "
                          "batch. 0 = full-vocab softmax (fine for small catalogs like "
                          "Beauty ~12k items). For large catalogs (100k+ items, e.g. "
                          "Electronics) set this to e.g. 200-500 or training will OOM "
                          "regardless of batch size, since full softmax cost scales "
                          "with vocab size.")
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--sasrec_ckpt", type=str, default=None,
                     help="path to save/reuse the pretrained SASRec checkpoint")
    ap.add_argument("--skip_pretrain", action="store_true",
                     help="load --sasrec_ckpt and skip stage 1")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"[cfg] {vars(args)}")

    # ---- Data ----
    t0 = time.time()
    seqs, n_users, n_items = fast_load_and_filter(
        args.reviews, min_inter=args.min_inter, max_users=args.max_users)
    print(f"[data] loaded in {time.time()-t0:.1f}s")
    train_seqs, val_data, test_data = prepare_loo(seqs)
    train_ds = SeqDataset(train_seqs, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=False)

    # ---- Model ----
    backbone = SASRecBackbone(num_items=n_items, d=args.d, H=args.H,
                              num_blocks=args.num_blocks, max_len=args.max_len,
                              dropout=args.dropout).to(device)
    sas = SASRecModel(backbone).to(device)
    print(f"[model] SASRec params: {sum(p.numel() for p in sas.parameters()):,}")

    # ---- Stage 1: pretrain SASRec ----
    if args.skip_pretrain and args.sasrec_ckpt and Path(args.sasrec_ckpt).exists():
        print(f"[stage1] loading pretrained SASRec from {args.sasrec_ckpt}")
        sas.load_state_dict(torch.load(args.sasrec_ckpt, map_location=device))
    else:
        print("[stage1] pretraining SASRec")
        run_stage("SASRec", sas, train_loader, val_data, test_data,
                  epochs=args.pretrain_epochs, patience=args.patience, lr=args.lr,
                  max_len=args.max_len, device=device,
                  grad_accum_steps=args.grad_accum_steps, use_amp=args.amp,
                  eval_batch=args.eval_batch, num_items=n_items, num_neg=args.num_neg)
        if args.sasrec_ckpt:
            torch.save(sas.state_dict(), args.sasrec_ckpt)
            print(f"[stage1] saved SASRec checkpoint to {args.sasrec_ckpt}")

    # ---- Stage 2: build FAME on top of pretrained backbone, finetune ----
    fame = FAMEModel(backbone, N=args.N, dropout=args.dropout).to(device)
    print(f"[model] FAME params: {sum(p.numel() for p in fame.parameters()):,}")
    print("[stage2] finetuning FAME end-to-end")
    run_stage("FAME", fame, train_loader, val_data, test_data,
              epochs=args.finetune_epochs, patience=args.patience, lr=args.lr,
              max_len=args.max_len, device=device,
              grad_accum_steps=args.grad_accum_steps, use_amp=args.amp,
              eval_batch=args.eval_batch, num_items=n_items, num_neg=args.num_neg)


if __name__ == "__main__":
    main()
