
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import random
from datetime import datetime
import sys
import os




SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

HIST_LEN = 50
EMB_DIM = 384
N_LAYERS = 2
N_HEADS = 2
DROPOUT = 0.2
CAND_SIZE = 100          
NUM_NEG = CAND_SIZE - 1

BATCH_SIZE = 256
EPOCHS = n_epochs
LR = 1e-3
TOPK = [1, 5, 10]

print("Device:", device)

USER_ITEMS_PATH = "/storage/TRM/steam_trm/user_items_mapped.pkl"

with open(USER_ITEMS_PATH, "rb") as f:
    user_items_raw = pickle.load(f)


def dedup_preserve_order(seq):
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
user_items_raw = {u: seq for u, seq in user_items_raw.items() if len(seq) >= 2}

# Determine item count directly from the data (no SBERT file involved)
num_items_real = max(i for seq in user_items_raw.values() for i in seq) + 1


user_items = {
    u: [i + 1 for i in seq]
    for u, seq in user_items_raw.items()
}
num_items = num_items_real + 1  # includes padding slot at index 0

print(f"Real items: {num_items_real}, Total (with pad): {num_items}")


def split_sequences(user_items):
    train, test = [], []
    for u, seq in user_items.items():
        if len(seq) < 2:
            continue
        train.append(seq[:-1])
        test.append((seq[:-1], seq[-1]))
    return train, test

train_seqs, test_seqs = split_sequences(user_items)
print(f"Train users: {len(train_seqs)}, Test users: {len(test_seqs)}")


def build_hist_tensor(hist_trunc, max_len=HIST_LEN):
    """Left-pad with 0 (the reserved padding index), mask by position."""
    L = len(hist_trunc)
    h_ids = torch.zeros(max_len, dtype=torch.long)
    h_mask = torch.zeros(max_len, dtype=torch.float32)
    if L > 0:
        trunc = hist_trunc[-max_len:]
        Lt = len(trunc)
        h_ids[-Lt:] = torch.tensor(trunc, dtype=torch.long)
        h_mask[-Lt:] = 1.0
    return h_ids, h_mask


class SASRecDataset(Dataset):
    def __init__(self, user_seqs):
        self.seqs = [s for s in user_seqs if len(s) >= 2]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        t = random.randint(1, len(seq) - 1)
        return seq[:t], seq[t]


def sample_negatives_batch(hist_sets, targets, num_items, num_neg):
    """Rejection-sample negatives per example, skipping padding index 0."""
    B = len(targets)
    all_negs = np.empty((B, num_neg), dtype=np.int64)
    for i in range(B):
        hset = hist_sets[i]
        tgt = targets[i]
        negs = []
        while len(negs) < num_neg:
            draw = np.random.randint(1, num_items, size=num_neg * 2)
            for n in draw:
                if n != tgt and n not in hset:
                    negs.append(n)
                    if len(negs) == num_neg:
                        break
        all_negs[i] = negs[:num_neg]
    return all_negs


def collate_full(batch):
    B = len(batch)
    h_ids = torch.zeros(B, HIST_LEN, dtype=torch.long)
    mask = torch.zeros(B, HIST_LEN, dtype=torch.float)
    tgt = torch.zeros(B, dtype=torch.long)
    hist_sets = []

    for i, (hist, t) in enumerate(batch):
        h_i, m_i = build_hist_tensor(hist, HIST_LEN)
        h_ids[i] = h_i
        mask[i] = m_i
        tgt[i] = t
        hist_sets.append(set(hist[-HIST_LEN:]))

    return h_ids, mask, tgt, hist_sets


train_loader = DataLoader(
    SASRecDataset(train_seqs),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_full
)


class SASRec(nn.Module):
    def __init__(self, num_items, dim, max_len, n_layers=2, n_heads=2, dropout=0.2):
        super().__init__()
        self.num_items = num_items
        self.dim = dim
        self.item_emb = nn.Embedding(num_items, dim, padding_idx=0)
        nn.init.xavier_uniform_(self.item_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].fill_(0)  # keep padding row zeroed after init

        self.pos_emb = nn.Embedding(max_len, dim)
        nn.init.normal_(self.pos_emb.weight, std=0.01)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=4 * dim,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

    def get_user_embedding(self, hist_ids, hist_mask):
        B, L = hist_ids.size()
        pos = torch.arange(L, device=hist_ids.device).unsqueeze(0)
        x = self.item_emb(hist_ids) + self.pos_emb(pos)
        key_padding_mask = (hist_mask == 0)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        idx = hist_mask.sum(dim=1).long() - 1
        idx = idx.clamp(min=0)
        return x[torch.arange(B, device=hist_ids.device), idx]

    def forward(self, hist_ids, hist_mask, cand_ids):
        seq_repr = self.get_user_embedding(hist_ids, hist_mask)
        c_emb = self.item_emb(cand_ids)
        logits = torch.einsum("bd,bnd->bn", seq_repr, c_emb)
        return logits


s_model = SASRec(num_items=num_items, dim=EMB_DIM, max_len=HIST_LEN,
                  n_layers=N_LAYERS, n_heads=N_HEADS, dropout=DROPOUT).to(device)
optimizer = torch.optim.Adam(s_model.parameters(), lr=LR)


@torch.no_grad()
def evaluate_sampled(model, test_seqs, ks=(1, 5, 10), num_candidates=CAND_SIZE):
    model.eval()
    hits = {k: 0 for k in ks}
    ndcgs = {k: 0 for k in ks}
    total = 0

    for hist, tgt in test_seqs:
        hist_trunc = hist[-HIST_LEN:]
        hist_set = set(hist)  # full history excluded

        negs = []
        while len(negs) < num_candidates - 1:
            n = random.randint(1, num_items - 1)  # skip padding idx 0
            if n != tgt and n not in hist_set:
                negs.append(n)
        candidates = negs + [tgt]
        random.shuffle(candidates)
        tgt_idx = candidates.index(tgt)

        h_ids, h_mask = build_hist_tensor(hist_trunc, HIST_LEN)
        h = h_ids.unsqueeze(0).to(device)
        m = h_mask.unsqueeze(0).to(device)
        c = torch.tensor(candidates, dtype=torch.long, device=device).unsqueeze(0)

        logits = model(h, m, c)[0]
        ranks = logits.argsort(descending=True)
        rank = (ranks == tgt_idx).nonzero(as_tuple=True)[0].item() + 1

        for k in ks:
            if rank <= k:
                hits[k] += 1
                ndcgs[k] += 1.0 / np.log2(rank + 1)
        total += 1

    return {
        **{f"HR@{k}": hits[k] / total for k in ks},
        **{f"NDCG@{k}": ndcgs[k] / total for k in ks},
    }



@torch.no_grad()
def evaluate_full_ranking(model, test_seqs, num_items, ks=(1, 5, 10),
                           batch_size=512, item_chunk_size=4096, max_len=HIST_LEN):
    model.eval()
    hits = {k: 0 for k in ks}
    ndcgs = {k: 0 for k in ks}
    precs = {k: 0 for k in ks}
    total = 0

    all_item_embs = model.item_emb.weight  # [num_items, D], includes padding row at idx 0

    for batch_start in range(0, len(test_seqs), batch_size):
        batch = test_seqs[batch_start: batch_start + batch_size]
        B = len(batch)

        h_ids_list, h_mask_list, targets_list, hist_sets = [], [], [], []
        for hist, tgt in batch:
            hist_trunc = hist[-max_len:]
            h_ids, h_mask = build_hist_tensor(hist_trunc, max_len)
            h_ids_list.append(h_ids)
            h_mask_list.append(h_mask)
            targets_list.append(tgt)
            hist_sets.append(set(hist))  # exclude FULL history, not just windowed

        h = torch.stack(h_ids_list).to(device)
        hm = torch.stack(h_mask_list).to(device)
        targets = torch.tensor(targets_list, dtype=torch.long, device=device)

        seq_repr = model.get_user_embedding(h, hm)  # [B, D]

        scores = torch.empty(B, num_items, device=device)
        for start in range(0, num_items, item_chunk_size):
            end = min(start + item_chunk_size, num_items)
            scores[:, start:end] = seq_repr @ all_item_embs[start:end].T

        # mask padding index 0 (never a valid recommendation)
        scores[:, 0] = float('-inf')

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



print("\nTraining...")
for epoch in range(EPOCHS):
    s_model.train()
    total_loss = 0

    for h, m, t, hist_sets in train_loader:
        h, m, t = h.to(device), m.to(device), t.to(device)
        B = h.size(0)

        negs_np = sample_negatives_batch(hist_sets, t.cpu().numpy(), num_items, NUM_NEG)
        pos_pos = np.random.randint(0, CAND_SIZE, size=B)

        cand_np = np.empty((B, CAND_SIZE), dtype=np.int64)
        for i in range(B):
            pp = pos_pos[i]
            cand_np[i, :pp] = negs_np[i, :pp]
            cand_np[i, pp] = t[i].item()
            cand_np[i, pp + 1:] = negs_np[i, pp:]

        c_tensor = torch.tensor(cand_np, dtype=torch.long, device=device)
        tgt_idx = torch.tensor(pos_pos, dtype=torch.long, device=device)

        logits = s_model(h, m, c_tensor)
        loss = F.cross_entropy(logits, tgt_idx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")

    


print("\nFinal Full-Ranking Evaluation:")
full_res = evaluate_full_ranking(s_model, test_seqs, num_items, TOPK)
print({k: round(v, 4) for k, v in full_res.items()})




import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import random
import copy
import math


seed = 37
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


HIST_LEN = 50
NUM_SUP = 2
NUM_INNER = 2
GAMMA = 0.7
TRM_EPOCHS = 3
BATCH_SIZE = 256
TRM_LR = 1e-3
TEMP = 0.07

CAND_SIZE = 100
NUM_NEG = CAND_SIZE - 1
TOPK = [1, 5, 10]

# =========================================================
# 3. Load Data
# =========================================================
USER_ITEMS_PATH = "/storage/TRM/steam_trm/user_items_mapped.pkl"
EMB_PATH = "/storage/TRM/steam_trm/steam_item_sbert.pt"

with open(USER_ITEMS_PATH, "rb") as f:
    user_items = pickle.load(f)

num_items = max(max(seq) for seq in user_items.values()) + 1
EMB_DIM = 384

item_embs_raw = torch.randn(num_items, EMB_DIM) * (1.0 / math.sqrt(EMB_DIM))
#item_embs_raw = torch.load(EMB_PATH, map_location="cpu")


valid_ids=set(range(item_embs_raw.shape[0]))
user_items = {
    u: [i for i in seq if i in valid_ids]
    for u, seq in user_items.items() 
}

user_items={u: seq for u, seq in user_items.items() if len(seq)>=2}
num_items=item_embs_raw.shape[0]

print(f"Items: {num_items}, Embedding dim: {EMB_DIM}")





print(f"item_embs_raw shape: {item_embs_raw.shape}")
print(f"num_items (from pkl): {num_items}")
assert item_embs_raw.shape[0] == num_items, (
    f"MISMATCH: embedding table has {item_embs_raw.shape[0]} rows "
    f"but item IDs go up to {num_items - 1} (num_items={num_items})"
)
assert item_embs_raw.shape[1] == EMB_DIM, (
    f"MISMATCH: embedding dim is {item_embs_raw.shape[1]}, expected {EMB_DIM}"
)


class TRM_Sequential(nn.Module):
    def __init__(self, item_embs, dim, num_sup, num_inner):
        super().__init__()

        self.item_embs = nn.Embedding.from_pretrained(item_embs, freeze=False)

        self.core = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )

        self.correction_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

        self.num_sup = num_sup
        self.num_inner = num_inner
        self.step_scale = 0.7  # stable recursion scale

    def forward(self, hist_ids, cand_ids, hist_mask):
        h_embs = self.item_embs(hist_ids)

        mask = hist_mask.unsqueeze(-1)
        x = (h_embs * mask).sum(1) / (mask.sum(1) + 1e-8)

        y = x.clone()
        z = torch.zeros_like(x)

        logits_history = []

        for _ in range(self.num_sup):

            # Inner recursion
            for _ in range(self.num_inner):
                z = self.core(torch.cat([x, y, z], dim=-1))

            # Correction gate
            gate = self.correction_gate(torch.cat([x, y], dim=-1))
            z = z * (1 - gate) + x * gate

            # Update
            delta = self.core(torch.cat([x, y, z], dim=-1))
            y = y + self.step_scale * torch.tanh(delta)

            # Score candidates
            c_embs = self.item_embs(cand_ids)
            logits = torch.einsum("bd,bnd->bn", y, c_embs)
            logits_history.append(logits)

        self.y_final = y.detach()
        return torch.stack(logits_history, dim=0)


def sample_candidates(hist_set, target):
    negatives = []
    while len(negatives) < NUM_NEG:
        neg = random.randint(0, num_items - 1)
        if neg != target and neg not in hist_set:
            negatives.append(neg)

    candidates = negatives + [target]
    random.shuffle(candidates)
    return candidates, candidates.index(target)

def split_sequences(user_items):
    train = []
    test = []

    for seq in user_items.values():
        train.append(seq[:-1])
        test.append((seq[:-1], seq[-1]))

    return train, test

train_seqs, test_seqs = split_sequences(user_items)


class TRMTrainDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = [seq for seq in sequences if len(seq) >= 2]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        t = np.random.randint(1, len(seq))
        hist = seq[:t]
        tgt = seq[t]
        return hist, tgt


class TRMTestDataset(Dataset):
    def __init__(self, test_pairs):
        self.test_pairs = test_pairs

    def __len__(self):
        return len(self.test_pairs)

    def __getitem__(self, idx):
        hist, tgt = self.test_pairs[idx]
        return hist, tgt




def collate_fn(batch):
    B = len(batch)

    h_ids = torch.zeros(B, HIST_LEN, dtype=torch.long)
    h_mask = torch.zeros(B, HIST_LEN, dtype=torch.float32)
    c_ids = torch.zeros(B, CAND_SIZE, dtype=torch.long)
    t_pos = torch.zeros(B, dtype=torch.long)

    for i, (hist, tgt) in enumerate(batch):
        hist = hist[-HIST_LEN:]
        L = len(hist)

        # --- DEFENSIVE CHECK 1: history ids ---
        for hid in hist:
            assert 0 <= hid < num_items, f"BAD HIST ID: {hid} (num_items={num_items}), user hist={hist}, target={tgt}"

        assert 0 <= tgt < num_items, f"BAD TARGET ID: {tgt} (num_items={num_items})"

        if L > 0:
            h_ids[i, -L:] = torch.tensor(hist, dtype=torch.long)
            h_mask[i, -L:] = 1.0

        candidates, tgt_idx = sample_candidates(set(hist), tgt)

        # --- DEFENSIVE CHECK 2: candidate ids ---
        for cid in candidates:
            assert 0 <= cid < num_items, f"BAD CANDIDATE ID: {cid} (num_items={num_items}), candidates={candidates}"

        c_ids[i] = torch.tensor(candidates, dtype=torch.long)
        t_pos[i] = tgt_idx

    return h_ids, c_ids, h_mask, t_pos


train_loader = DataLoader(
    TRMTrainDataset(train_seqs),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    TRMTestDataset(test_seqs),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)


# =========================================================
# 7. EMA
# =========================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)


def collate_fn_full(batch):
    """Test collate for full-ranking: keep history for exclusion, target as a single ID."""
    B = len(batch)
    h_ids = torch.zeros(B, HIST_LEN, dtype=torch.long)
    h_mask = torch.zeros(B, HIST_LEN, dtype=torch.float32)
    targets = torch.zeros(B, dtype=torch.long)
    hist_sets = []  # list of sets, for masking seen items

    for i, (hist, tgt) in enumerate(batch):
        hist_trunc = hist[-HIST_LEN:]
        L = len(hist_trunc)
        if L > 0:
            h_ids[i, -L:] = torch.tensor(hist_trunc, dtype=torch.long)
            h_mask[i, -L:] = 1.0
        targets[i] = tgt
        hist_sets.append(set(hist))  # exclude ALL seen items, not just windowed

    return h_ids, h_mask, targets, hist_sets


test_loader_full = DataLoader(
    TRMTestDataset(test_seqs),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn_full
)


@torch.no_grad()
def evaluate(model, loader, ks):
    model.eval()
    hits = {k: 0 for k in ks}
    ndcgs = {k: 0 for k in ks}
    precs = {k: 0 for k in ks}
    total = 0

    # Benchmark tools
    total_latency_ms = 0.0
    is_cuda = device.type == "cuda"

    for h, c, m, t in loader:
        h, c, m, t = h.to(device), c.to(device), m.to(device), t.to(device)
        
        # Accurate GPU timing requires synchronization events
        if is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            import time
            start_time = time.time()

        # --- Inference Pass ---
        logits = model(h, c, m)[-1]
        # ----------------------

        if is_cuda:
            end_event.record()
            torch.cuda.synchronize()  # Wait for GPU to finish execution
            batch_latency = start_event.elapsed_time(end_event) # Returns milliseconds
        else:
            batch_latency = (time.time() - start_time) * 1000.0 # Convert seconds to ms

        total_latency_ms += batch_latency

        # Metric calculation
        ranks = logits.argsort(dim=1, descending=True)
        for i in range(h.size(0)):
            rank = (ranks[i] == t[i]).nonzero(as_tuple=True)[0].item() + 1
            for k in ks:
                if rank <= k:
                    hits[k] += 1
                    ndcgs[k] += 1 / math.log2(rank + 1)
                    precs[k] += 1 / k
        total += h.size(0)

    # Calculate final latency metrics
    avg_latency_per_user = total_latency_ms / total

    return {
        **{f"HR@{k}": hits[k]/total for k in ks},
        **{f"NDCG@{k}": ndcgs[k]/total for k in ks},
        **{f"Prec@{k}": precs[k]/total for k in ks},
        "Total_Latency_ms": total_latency_ms,
        "Avg_Latency_Per_User_ms": avg_latency_per_user
    }



@torch.no_grad()
def evaluate_full_ranking(model, loader, ks, num_items, chunk_size=4096):
    """
    Full-ranking evaluation: score every item in the catalog, mask history,
    rank target against all remaining items.

    chunk_size: how many items to score per matmul chunk. Increase if you
                have GPU memory; decrease if you OOM on large catalogs.
    """
    model.eval()
    hits = {k: 0 for k in ks}
    ndcgs = {k: 0 for k in ks}
    precs = {k: 0 for k in ks}
    total = 0
    total_latency_ms = 0.0
    is_cuda = device.type == "cuda"

    # Grab the full item embedding matrix once (updated per-epoch since embeddings train)
    all_item_embs = model.item_embs.weight  # [num_items, d]

    for h, m, targets, hist_sets in loader:
        h = h.to(device)
        m = m.to(device)
        targets = targets.to(device)
        B = h.size(0)

        if is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            import time
            start_time = time.time()

        # --- Compute final preference vector y_T via the model's recursion ---
        # We reuse the model's forward loop but score against ALL items.
        h_embs = model.item_embs(h)
        mask_ = m.unsqueeze(-1)
        x = (h_embs * mask_).sum(1) / (mask_.sum(1) + 1e-8)

        y = x.clone()
        z = torch.zeros_like(x)

        for _ in range(model.num_sup):
            for _ in range(model.num_inner):
                z = model.core(torch.cat([x, y, z], dim=-1))
            gate = model.correction_gate(torch.cat([x, y], dim=-1))
            z = z * (1 - gate) + x * gate
            delta = model.core(torch.cat([x, y, z], dim=-1))
            y = y + model.step_scale * torch.tanh(delta)
        # y is [B, d] — the final preference state

        # --- Score against all items in chunks to control memory ---
        scores = torch.empty(B, num_items, device=device)
        for start in range(0, num_items, chunk_size):
            end = min(start + chunk_size, num_items)
            scores[:, start:end] = y @ all_item_embs[start:end].T

        # --- Mask out items already in each user's history ---
        # Set their scores to -inf so they never appear in top-k
        for i, hset in enumerate(hist_sets):
            if len(hset) > 0:
                idx = torch.tensor(list(hset), device=device, dtype=torch.long)
                scores[i, idx] = float('-inf')

        # --- Compute rank of target for each user ---
        # rank = 1 + (# items scoring strictly higher than the target)
        target_scores = scores.gather(1, targets.unsqueeze(1))  # [B, 1]
        ranks = (scores > target_scores).sum(dim=1) + 1  # [B]

        if is_cuda:
            end_event.record()
            torch.cuda.synchronize()
            batch_latency = start_event.elapsed_time(end_event)
        else:
            batch_latency = (time.time() - start_time) * 1000.0
        total_latency_ms += batch_latency

        # --- Accumulate metrics ---
        ranks_cpu = ranks.cpu().numpy()
        for r in ranks_cpu:
            for k in ks:
                if r <= k:
                    hits[k] += 1
                    ndcgs[k] += 1.0 / math.log2(r + 1)
                    precs[k] += 1.0 / k
        total += B

    

    return {
        **{f"HR@{k}": hits[k] / total for k in ks},
        **{f"NDCG@{k}": ndcgs[k] / total for k in ks},
        **{f"Prec@{k}": precs[k] / total for k in ks},
        
    }

# =========================================================
# 9. Training
# =========================================================
model = TRM_Sequential(item_embs_raw.to(device), EMB_DIM, NUM_SUP, NUM_INNER).to(device)
ema = EMA(model)

optimizer = torch.optim.Adam(model.parameters(), lr=TRM_LR)
criterion = nn.CrossEntropyLoss()



all_ids = []
for u, seq in user_items.items():
    all_ids.extend(seq)

all_ids = np.array(all_ids)
print(f"Total interactions: {len(all_ids)}")
print(f"Min ID: {all_ids.min()}")
print(f"Max ID: {all_ids.max()}")
print(f"num_items (embedding table size): {num_items}")
print(f"Unique IDs: {len(np.unique(all_ids))}")
print(f"Any ID >= num_items? {(all_ids >= num_items).any()}")
print(f"Any negative ID? {(all_ids < 0).any()}")
print(f"Dtype: {all_ids.dtype}")

# Check a few sequences
for u in list(user_items.keys())[:3]:
    seq = user_items[u]
    print(f"User {u}: len={len(seq)}, min={min(seq)}, max={max(seq)}, sample={seq[:5]}")

print("\nTraining...")

for epoch in range(TRM_EPOCHS):
    model.train()

    for h, c, m, t in train_loader:
        h, c, m, t = h.to(device), c.to(device), m.to(device), t.to(device)
        logits = model(h, c, m)

        loss = sum(
            (GAMMA ** (NUM_SUP - 1 - k)) *
            criterion(logits[k] / TEMP, t)
            for k in range(NUM_SUP)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model)

    #metrics = evaluate(ema.model, test_loader, TOPK)
    metrics = evaluate_full_ranking(ema.model, test_loader_full, TOPK, num_items, chunk_size=4096)


    print(f"Epoch {epoch+1} | " +
          " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

print("\nTraining complete.")
