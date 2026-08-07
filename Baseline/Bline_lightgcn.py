
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import random
from collections import defaultdict
from datetime import datetime
import os

SEED = 37
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

EMB_DIM = 384
N_LAYERS = 3
BATCH_SIZE = 2048
EPOCHS = num_epochs
LR = 1e-3
NUM_NEG = 100
TOPK = [1, 10]


USER_ITEMS_PATH = "/storage/TRM/steam_trm/user_items_mapped.pkl"

with open(USER_ITEMS_PATH, "rb") as f:
    user_items = pickle.load(f)

# re-index users
user_ids = list(user_items.keys())
uid_map = {u: i for i, u in enumerate(user_ids)}

user_items = {
    uid_map[u]: list(set(seq))
    for u, seq in user_items.items()
    if len(seq) >= 2
}

num_users = len(user_items)
num_items = max(i for seq in user_items.values() for i in seq) + 1



# Re-index items
all_items = sorted({i for seq in user_items.values() for i in seq})
iid_map = {i: idx for idx, i in enumerate(all_items)}

user_items = {
    u: [iid_map[i] for i in seq]
    for u, seq in user_items.items()
}

num_items = len(iid_map)
# ----- USER REMAP -----
raw_users = sorted(user_items.keys())
uid_map = {u: idx for idx, u in enumerate(raw_users)}

# ----- ITEM REMAP -----
raw_items = sorted({i for seq in user_items.values() for i in seq})
iid_map = {i: idx for idx, i in enumerate(raw_items)}

# ----- APPLY REMAP -----
user_items = {
    uid_map[u]: [iid_map[i] for i in seq]
    for u, seq in user_items.items()
}

num_users = len(uid_map)
num_items = len(iid_map)



train_interactions = []
test_data = []

for u, seq in user_items.items():
    train_seq = seq[:-1]
    test_item = seq[-1]
    test_data.append((u, train_seq, test_item))
    for i in train_seq:
        train_interactions.append((u, i))

print(len(train_interactions))

def build_norm_adj(train_data, num_users, num_items):
    rows, cols = [], []
    for u, i in train_data:
        rows += [u, i + num_users]
        cols += [i + num_users, u]

    idx = torch.LongTensor([rows, cols])
    vals = torch.ones(len(rows))

    size = num_users + num_items
    adj = torch.sparse_coo_tensor(idx, vals, (size, size))

    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

    r, c = idx
    norm_vals = deg_inv_sqrt[r] * deg_inv_sqrt[c]
    norm_adj = torch.sparse_coo_tensor(idx, norm_vals, (size, size))

    return norm_adj.coalesce().to(device)

norm_adj = build_norm_adj(train_interactions, num_users, num_items)


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, norm_adj):
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight])
        embs = [all_emb]

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(norm_adj, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs).mean(dim=0)
        return embs[:num_users], embs[num_users:]



class BPRDataset(Dataset):
    def __init__(self, user_items, num_items):
        self.user_items = user_items
        self.num_items = num_items
        # Flatten to (user, pos_item) pairs — one entry per interaction
        self.pairs = [(u, i) for u, seq in user_items.items() for i in seq]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        u, pos = self.pairs[idx]
        user_set = self.user_items[u]  # keep as a set for O(1) lookup, see Bug 3

        while True:
            neg = random.randint(0, self.num_items - 1)
            if neg not in user_set:
                break

        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )



model = LightGCN(num_users, num_items, EMB_DIM, N_LAYERS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_user_items = {
    u: seq[:-1]
    for u, seq in user_items.items()
    if len(seq) >= 2
}

train_loader = DataLoader(
    BPRDataset(train_user_items, num_items),
    batch_size=BATCH_SIZE,
    shuffle=True
)


@torch.no_grad()
def evaluate_full_ranking_lightgcn(
    model,
    test_data,
    norm_adj,
    num_items,
    ks=(1, 5, 10),
    batch_size=512,
    item_chunk_size=4096
):
    """
    Full-ranking evaluation for LightGCN: score every item in the catalog
    (excluding the user's full interaction history), rank the target
    against all remaining items.

    Batches users together for efficiency instead of the original
    per-user Python loop.
    """
    model.eval()
    user_emb, item_emb = model(norm_adj)  # compute embeddings ONCE, not per-user

    hits = {k: 0 for k in ks}
    ndcgs = {k: 0 for k in ks}
    precs = {k: 0 for k in ks}
    recalls = {k: 0 for k in ks}
    accuracy = 0.0
    total = 0

    # Process test users in batches
    for batch_start in range(0, len(test_data), batch_size):
        batch = test_data[batch_start: batch_start + batch_size]
        B = len(batch)

        user_ids = torch.tensor([u for u, _, _ in batch], dtype=torch.long, device=device)
        targets = torch.tensor([tgt for _, _, tgt in batch], dtype=torch.long, device=device)
        hist_sets = [set(hist) for _, hist, _ in batch]  # FULL history, not windowed

        u_emb_batch = user_emb[user_ids]  # [B, d]

        # --- Score against all items in chunks to control memory ---
        scores = torch.empty(B, num_items, device=device)
        for start in range(0, num_items, item_chunk_size):
            end = min(start + item_chunk_size, num_items)
            scores[:, start:end] = u_emb_batch @ item_emb[start:end].T

        # --- Mask out items already in each user's history ---
        for i, hset in enumerate(hist_sets):
            if len(hset) > 0:
                idx = torch.tensor(list(hset), device=device, dtype=torch.long)
                scores[i, idx] = float('-inf')

        # --- Compute rank of target for each user ---
        target_scores = scores.gather(1, targets.unsqueeze(1))  # [B, 1]
        ranks = (scores > target_scores).sum(dim=1) + 1  # [B]

        ranks_cpu = ranks.cpu().numpy()
        for r in ranks_cpu:
            for k in ks:
                if r <= k:
                    hits[k] += 1
                    recalls[k] += 1
                    precs[k] += 1.0 / k
                    ndcgs[k] += 1.0 / np.log2(r + 1)
            if r == 1:
                accuracy += 1

        total += B

    metrics = {
        "HR": {k: hits[k] / total for k in ks},
        "NDCG": {k: ndcgs[k] / total for k in ks},
        "Precision": {k: precs[k] / total for k in ks},
        "Recall": {k: recalls[k] / total for k in ks},
        "Accuracy": accuracy / total,
    }
    return metrics
def bpr_loss(u, p, n):
    return -torch.log(torch.sigmoid(
        (u * p).sum(dim=1) - (u * n).sum(dim=1)
    )).mean()

user_emb, item_emb = model(norm_adj)



assert user_emb.size(0) == num_users
assert item_emb.size(0) == num_items

# dataset sanity
for u, seq in user_items.items():
    assert u < num_users
    for i in seq:
        assert 0 <= i < num_items


print("\nTraining LightGCN...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for u, p, n in train_loader:
        u, p, n = u.to(device), p.to(device), n.to(device)

        #    recompute embeddings EVERY step
        user_emb, item_emb = model(norm_adj)

        loss = bpr_loss(
            user_emb[u],
            item_emb[p],
            item_emb[n]
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss {total_loss / len(train_loader):.4f}")

    res = evaluate_full_ranking_lightgcn(model, test_data,     norm_adj, num_items, ks=(1, 5, 10))
    print(res)
