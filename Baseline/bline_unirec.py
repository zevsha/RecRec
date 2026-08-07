import os
import random
import math
import pickle
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


SEED = 37
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


EMB_DIM = 384
MAX_LEN = 50
N_LAYERS = 3
N_HEADS = 2
DROPOUT = 0.2

BATCH_SIZE = 2048
EPOCHS = n_epochs
LR = 1e-3

NUM_NEG = 100
CAND_EVAL = 100
UNIFORM_K = 256
UNIFORM_LAMBDA = 0.1
FREQ_ALPHA = 0.5

PRINT_EVERY = 1


USER_ITEMS_PATH = "/storage/TRM/steam_trm/user_items_mapped.pkl"

with open(USER_ITEMS_PATH, "rb") as f:
    user_items_raw = pickle.load(f)


def dedup_preserve_order(seq):
    """Remove repeat interactions with the same item while keeping
    chronological order intact (do NOT use set() for this — it destroys order)."""
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


user_items_raw = {
    u: dedup_preserve_order(seq)
    for u, seq in user_items_raw.items()
    if len(seq) >= 2
}

# ----- Re-index users -----
raw_users = sorted(user_items_raw.keys())
uid_map = {u: idx for idx, u in enumerate(raw_users)}

# ----- Re-index items -----
raw_items = sorted({i for seq in user_items_raw.values() for i in seq})
iid_map = {i: idx for idx, i in enumerate(raw_items)}

# ----- Apply remaps -----
user_items = {
    uid_map[u]: [iid_map[i] for i in seq]
    for u, seq in user_items_raw.items()
}

num_users = len(uid_map)
num_items = len(iid_map)



all_ids = np.array([i for seq in user_items.values() for i in seq])
assert all_ids.min() >= 0, "Negative item ID found"
assert all_ids.max() < num_items, "Item ID out of bounds"

test_data = []
for u, seq in user_items.items():
    hist = seq[:-1]
    tgt = seq[-1]
    if len(hist) == 0:
        continue
    test_data.append((u, hist, tgt))



def compute_item_freqs(user_items):
    counts = Counter()
    for u, seq in user_items.items():
        
        for it in seq[:-1]:
            if isinstance(it, (list, tuple)):
                it = int(it[0])
            counts[int(it)] += 1
    return counts


item_freqs = compute_item_freqs(user_items)   # Counter {item: freq}
default_freq = 1.0

freq_arr = np.ones(num_items, dtype=float) * default_freq
for it, f in item_freqs.items():
    if it < num_items:
        freq_arr[it] = float(f)

freq_weights = (1.0 / (freq_arr ** FREQ_ALPHA))
freq_weights = freq_weights / freq_weights.mean()


def build_hist_tensor(hist_trunc, max_len=MAX_LEN):
    """Left-pad hist_trunc to max_len, return (ids, mask) with mask
    based on POSITION, not value — item ID 0 is a valid item, not padding."""
    L = len(hist_trunc)
    h_ids = torch.zeros(max_len, dtype=torch.long)
    h_mask = torch.zeros(max_len, dtype=torch.float32)
    if L > 0:
        trunc = hist_trunc[-max_len:]
        Lt = len(trunc)
        h_ids[-Lt:] = torch.tensor(trunc, dtype=torch.long)
        h_mask[-Lt:] = 1.0
    return h_ids, h_mask


class SeqTrainDataset(Dataset):
    def __init__(self, user_seqs):
        # user_seqs: list of sequences (list of ints); we use seq[:-1] as train prefixes
        self.seqs = [s for s in user_seqs if len(s) >= 2]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        # choose a random cut t in [1, len(seq)-1]
        t = random.randint(1, len(seq) - 1)
        hist = seq[:t]
        tgt = seq[t]
        return hist, tgt


def collate_train(batch, max_len=MAX_LEN, num_neg=NUM_NEG):
    B = len(batch)
    h_ids = torch.zeros(B, max_len, dtype=torch.long)
    h_mask = torch.zeros(B, max_len, dtype=torch.float32)
    cand_ids = torch.zeros(B, num_neg + 1, dtype=torch.long)
    tgt_idx = torch.zeros(B, dtype=torch.long)

    for i, (hist, tgt) in enumerate(batch):
        flat = []
        for h in hist:
            if isinstance(h, (list, tuple)):
                flat.append(int(h[0]))
            else:
                flat.append(int(h))

        h_ids_i, h_mask_i = build_hist_tensor(flat, max_len)
        h_ids[i] = h_ids_i
        h_mask[i] = h_mask_i

        hist_trunc = flat[-max_len:]
        hist_set = set(hist_trunc)

        negs = []
        while len(negs) < num_neg:
            n = random.randint(0, num_items - 1)
            if n != tgt and n not in hist_set:
                negs.append(n)

        pos_pos = random.randint(0, num_neg)
        cand = negs[:pos_pos] + [int(tgt)] + negs[pos_pos:]
        cand_ids[i] = torch.tensor(cand, dtype=torch.long)
        tgt_idx[i] = pos_pos

    return h_ids, h_mask, cand_ids, tgt_idx


train_seqs = [seq[:-1] for seq in user_items.values()]
train_dataset = SeqTrainDataset(train_seqs)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=lambda b: collate_train(b, MAX_LEN, NUM_NEG)
)


class UniRec(nn.Module):
    def __init__(self, num_items, emb_dim=EMB_DIM, max_len=MAX_LEN, n_layers=N_LAYERS, n_heads=N_HEADS, dropout=DROPOUT):
        super().__init__()
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.max_len = max_len

        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.xavier_uniform_(self.item_emb.weight)

        self.pos_emb = nn.Embedding(max_len, emb_dim)
        nn.init.normal_(self.pos_emb.weight, std=0.01)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=4 * emb_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim)
        )

    def forward_user_repr(self, hist_ids, hist_mask):
        B, L = hist_ids.shape
        pos = torch.arange(L, device=hist_ids.device).unsqueeze(0)
        x = self.item_emb(hist_ids) + self.pos_emb(pos)
        key_padding_mask = (hist_mask == 0)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [B, L, D]
        idx = hist_mask.sum(dim=1).long() - 1
        idx = idx.clamp(min=0)
        seq_repr = x[torch.arange(B, device=hist_ids.device), idx]
        seq_repr = self.proj(seq_repr)
        return seq_repr

    def score(self, seq_repr, cand_ids):
        c_emb = self.item_emb(cand_ids)
        scores = torch.einsum("bd,bcd->bc", seq_repr, c_emb)
        return scores

    def uniformity_loss(self, k=UNIFORM_K):
        with torch.no_grad():
            n = self.num_items
            if n <= k:
                idx = torch.arange(n, device=self.item_emb.weight.device)
            else:
                idx = torch.from_numpy(np.random.choice(n, k, replace=False)).to(self.item_emb.weight.device)
        sampled = self.item_emb.weight[idx]
        normed = F.normalize(sampled, p=2, dim=1)
        sims = torch.matmul(normed, normed.t())
        pairwise_sq = 2 - 2 * sims
        mask = ~torch.eye(pairwise_sq.size(0), dtype=torch.bool, device=pairwise_sq.device)
        vals = pairwise_sq[mask]
        if vals.numel() == 0:
            return torch.tensor(0., device=pairwise_sq.device)
        t = 2.0
        return torch.log(torch.mean(torch.exp(-t * vals)))


model = UniRec(num_items=num_items, emb_dim=EMB_DIM, max_len=MAX_LEN,
                   n_layers=N_LAYERS, n_heads=N_HEADS, dropout=DROPOUT).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)

freq_weights_t = torch.tensor(freq_weights, dtype=torch.float32, device=device)



def train_one_epoch(model, loader, opt, epoch_idx):
    model.train()
    total_loss = 0.0
    total_batches = 0
    for b_idx, (h, hm, cands, tgt_idx) in enumerate(loader):
        h = h.to(device)
        hm = hm.to(device)
        cands = cands.to(device)
        tgt_idx = tgt_idx.to(device)

        seq_repr = model.forward_user_repr(h, hm)
        logits = model.score(seq_repr, cands)

        ce_loss = F.cross_entropy(logits / 0.07, tgt_idx, reduction="none")
        pos_items = cands[torch.arange(cands.size(0)), tgt_idx]
        weights = freq_weights_t[pos_items]
        weighted_ce = (ce_loss * weights).mean()

        u_loss = model.uniformity_loss(k=UNIFORM_K)

        loss = weighted_ce + UNIFORM_LAMBDA * u_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        total_batches += 1

    avg = total_loss / max(1, total_batches)
    print(f"[Train] Epoch {epoch_idx+1} avg loss: {avg:.6f}")
    return avg


@torch.no_grad()
def evaluate(model, test_data, ks=(1, 10), num_candidates=CAND_EVAL):
    model.eval()
    hits = {k: 0 for k in ks}
    ndcgs = {k: 0 for k in ks}
    precs = {k: 0 for k in ks}
    total = 0

    for (u, hist, tgt) in test_data:
        if len(hist) == 0:
            continue

        hist_trunc = hist[-MAX_LEN:]
        hist_set = set(hist_trunc)
        candidates = [tgt]
        while len(candidates) < num_candidates:
            neg = random.randint(0, num_items - 1)
            if neg not in hist_set and neg != tgt:
                candidates.append(neg)
        seen = set(); cul = []
        for x in candidates:
            if x not in seen:
                seen.add(x); cul.append(x)
        candidates = cul
        if tgt not in candidates:
            candidates.append(tgt)

        h_ids, h_mask = build_hist_tensor(hist_trunc, MAX_LEN)
        h = h_ids.unsqueeze(0).to(device)
        hm = h_mask.unsqueeze(0).to(device)
        c = torch.tensor(candidates, dtype=torch.long, device=device).unsqueeze(0)

        seq_repr = model.forward_user_repr(h, hm)
        logits = model.score(seq_repr, c)[0]
        ranking = logits.argsort(descending=True)
        tgt_pos = (torch.tensor(candidates, device=device) == tgt).nonzero(as_tuple=True)[0].item()

        rank = (ranking == tgt_pos).nonzero(as_tuple=True)[0].item() + 1

        for k in ks:
            if rank <= k:
                hits[k] += 1
                ndcgs[k] += 1.0 / np.log2(rank + 1)
                precs[k] += 1.0 / k

        total += 1

    results = {}
    for k in ks:
        results[f"HR@{k}"] = hits[k] / total
        results[f"NDCG@{k}"] = ndcgs[k] / total
        results[f"Prec@{k}"] = precs[k] / total

    return results


@torch.no_grad()
def evaluate_full_ranking(model, test_data, num_items, ks=(1, 5, 10),
                           batch_size=512, item_chunk_size=4096, max_len=MAX_LEN):
    """
    Full-ranking evaluation: score every item in the catalog (excluding
    the user's full interaction history), rank the target against all
    remaining items. Batched for efficiency.
    """
    model.eval()
    hits = {k: 0 for k in ks}
    ndcgs = {k: 0 for k in ks}
    precs = {k: 0 for k in ks}
    total = 0

    all_item_embs = model.item_emb.weight  # [num_items, D]

    for batch_start in range(0, len(test_data), batch_size):
        batch = test_data[batch_start: batch_start + batch_size]
        batch = [(u, hist, tgt) for (u, hist, tgt) in batch if len(hist) > 0]
        if len(batch) == 0:
            continue
        B = len(batch)

        h_ids_list = []
        h_mask_list = []
        targets_list = []
        hist_sets = []

        for u, hist, tgt in batch:
            hist_trunc = hist[-max_len:]
            h_ids, h_mask = build_hist_tensor(hist_trunc, max_len)
            h_ids_list.append(h_ids)
            h_mask_list.append(h_mask)
            targets_list.append(tgt)
            hist_sets.append(set(hist))  # FULL history for exclusion, not windowed

        h = torch.stack(h_ids_list).to(device)
        hm = torch.stack(h_mask_list).to(device)
        targets = torch.tensor(targets_list, dtype=torch.long, device=device)

        seq_repr = model.forward_user_repr(h, hm)  # [B, D]

        scores = torch.empty(B, num_items, device=device)
        for start in range(0, num_items, item_chunk_size):
            end = min(start + item_chunk_size, num_items)
            scores[:, start:end] = seq_repr @ all_item_embs[start:end].T

        for i, hset in enumerate(hist_sets):
            if len(hset) > 0:
                idx = torch.tensor(list(hset), device=device, dtype=torch.long)
                scores[i, idx] = float('-inf')

        target_scores = scores.gather(1, targets.unsqueeze(1))
        ranks = (scores > target_scores).sum(dim=1) + 1

        ranks_cpu = ranks.cpu().numpy()
        for r in ranks_cpu:
            for k in ks:
                if r <= k:
                    hits[k] += 1
                    ndcgs[k] += 1.0 / np.log2(r + 1)
                    precs[k] += 1.0 / k
        total += B

    return {
        **{f"HR@{k}": hits[k] / total for k in ks},
        **{f"NDCG@{k}": ndcgs[k] / total for k in ks},
        **{f"Prec@{k}": precs[k] / total for k in ks},
    }


print("starting training")
for epoch in range(EPOCHS):
    train_one_epoch(model, train_loader, opt, epoch)



print("\nFull-ranking evaluation:")
final_res_full = evaluate_full_ranking(model, test_data, num_items, ks=(1, 5, 10))
print(final_res_full)
