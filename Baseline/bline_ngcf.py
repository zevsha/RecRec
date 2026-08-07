
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from collections import defaultdict
import math

SEED = 107
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device:", device)

EMB_DIM = 384
N_LAYERS = 3
BATCH_SIZE = 1024
EPOCHS =n_epochs
LR = 1e-3
TOPK = [1, 5, 10]
NUM_CANDIDATES = 100
HIST_LEN = 50

import pickle
from collections import defaultdict

USER_ITEMS_PATH = "/storage/TRM/steam_trm/user_items_mapped.pkl"

with open(USER_ITEMS_PATH, "rb") as f:
    raw_user_items = pickle.load(f)

raw_user_items = {
    u: list(set(seq))
    for u, seq in raw_user_items.items()
    if len(seq) >= 2
}

raw_users = sorted(raw_user_items.keys())
uid_map = {u: idx for idx, u in enumerate(raw_users)}

raw_items = sorted({i for seq in raw_user_items.values() for i in seq})
iid_map = {i: idx for idx, i in enumerate(raw_items)}

user_items = {
    uid_map[u]: [iid_map[i] for i in seq]
    for u, seq in raw_user_items.items()
}

num_users = len(uid_map)
num_items = len(iid_map)


HIST_LEN = 50

train_user_items = {}
train_interactions = []
test_data = []

for u, seq in user_items.items():
    seq = seq[:]
    train_seq = seq[:-1]
    test_item = seq[-1]

    train_user_items[u] = train_seq
    test_data.append((u, train_seq[-HIST_LEN:], test_item))

    for i in train_seq:
        train_interactions.append((u, i))


def build_norm_adj(train_edges, num_users, num_items):
    rows, cols = [], []

    for u, i in train_edges:
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

    return torch.sparse_coo_tensor(
        idx, norm_vals, (size, size)
    ).coalesce().to(device)

norm_adj = build_norm_adj(train_interactions, num_users, num_items)

class NGCF(nn.Module):
    def __init__(self, num_users, num_items, dim, n_layers, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers

        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        self.W_gc = nn.ModuleList()
        self.W_bi = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(n_layers):
            self.W_gc.append(nn.Linear(dim, dim))
            self.W_bi.append(nn.Linear(dim, dim))
            self.norms.append(nn.LayerNorm(dim))

        self.dropout = dropout

    def forward(self, norm_adj):
        all_emb = torch.cat(
            [self.user_emb.weight, self.item_emb.weight], dim=0
        )
        embs = [all_emb]

        for k in range(self.n_layers):
            side = torch.sparse.mm(norm_adj, all_emb)
            sum_emb = self.W_gc[k](side)
            bi_emb = self.W_bi[k](all_emb * side)

            all_emb = F.leaky_relu(sum_emb + bi_emb, 0.2)
            all_emb = self.norms[k](all_emb)
            all_emb = F.dropout(all_emb, self.dropout, training=self.training)

            embs.append(all_emb)

        final = torch.mean(torch.stack(embs), dim=0)
        return final[:num_users], final[num_users:]

class BPRDataset(Dataset):
    def __init__(self, interactions, user_items, num_items):
        self.data = interactions
        self.user_items = user_items
        self.num_items = num_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        u, pos = self.data[idx]
        neg = random.randint(0, self.num_items - 1)
        while neg in self.user_items[u]:
            neg = random.randint(0, self.num_items - 1)
        return u, pos, neg

def bpr_loss(u, p, n):
    return -torch.log(torch.sigmoid(
        (u * p).sum(dim=1) - (u * n).sum(dim=1)
    )).mean()


@torch.no_grad()
def evaluate(model, test_data):
    model.eval()
    user_emb, item_emb = model(norm_adj)

    metrics = {k: 0.0 for k in TOPK}
    ndcg = {k: 0.0 for k in TOPK}
    precision = {k: 0.0 for k in TOPK}
    recall = {k: 0.0 for k in TOPK}
    accuracy = 0.0

    total = 0

    for u, hist, tgt in test_data:
        hist = set(hist)
        candidates = [tgt]

        while len(candidates) < NUM_CANDIDATES:
            neg = random.randint(0, num_items - 1)
            if neg not in hist and neg != tgt:
                candidates.append(neg)

        scores = torch.matmul(
            user_emb[u],
            item_emb[candidates].t()
        )

        rank = scores.argsort(descending=True)
        tgt_idx = candidates.index(tgt)
        pos = (rank == tgt_idx).nonzero(as_tuple=True)[0].item() + 1

        if pos == 1:
            accuracy += 1

        for k in TOPK:
            if pos <= k:
                metrics[k] += 1
                recall[k] += 1
                precision[k] += 1 / k
                ndcg[k] += 1 / math.log2(pos + 1)

        total += 1

    results = {}
    for k in TOPK:
        results[f"HR@{k}"] = metrics[k] / total
        results[f"NDCG@{k}"] = ndcg[k] / total
        results[f"Precision@{k}"] = precision[k] / total
        results[f"Recall@{k}"] = recall[k] / total

    results["Accuracy"] = accuracy / total
    return results



@torch.no_grad()
def evaluate_full_ranking(model, test_data, norm_adj, num_items, ks=TOPK,
                           batch_size=512, item_chunk_size=4096):
    """
    Full-ranking evaluation: score every item in the catalog (excluding
    the user's full interaction history), rank the target against all
    remaining items. Batched for efficiency.
    """
    model.eval()
    user_emb, item_emb = model(norm_adj)  # compute embeddings ONCE

    hits = {k: 0 for k in ks}
    ndcgs = {k: 0 for k in ks}
    precision = {k: 0 for k in ks}
    recall = {k: 0 for k in ks}
    accuracy = 0.0
    total = 0

    for batch_start in range(0, len(test_data), batch_size):
        batch = test_data[batch_start: batch_start + batch_size]
        B = len(batch)

        user_ids = torch.tensor([u for u, _, _ in batch], dtype=torch.long, device=device)
        targets = torch.tensor([tgt for _, _, tgt in batch], dtype=torch.long, device=device)
        hist_sets = [set(hist) for _, hist, _ in batch]  # note: hist here is already
                                                            # train_seq[-HIST_LEN:] from
                                                            # test_data construction

        u_emb_batch = user_emb[user_ids]  # [B, d]

        scores = torch.empty(B, num_items, device=device)
        for start in range(0, num_items, item_chunk_size):
            end = min(start + item_chunk_size, num_items)
            scores[:, start:end] = u_emb_batch @ item_emb[start:end].T

        for i, hset in enumerate(hist_sets):
            if len(hset) > 0:
                idx = torch.tensor(list(hset), device=device, dtype=torch.long)
                scores[i, idx] = float('-inf')

        target_scores = scores.gather(1, targets.unsqueeze(1))
        ranks = (scores > target_scores).sum(dim=1) + 1  # [B]

        ranks_cpu = ranks.cpu().numpy()
        for r in ranks_cpu:
            for k in ks:
                if r <= k:
                    hits[k] += 1
                    recall[k] += 1
                    precision[k] += 1.0 / k
                    ndcgs[k] += 1.0 / math.log2(r + 1)
            if r == 1:
                accuracy += 1

        total += B

    results = {}
    for k in ks:
        results[f"HR@{k}"] = hits[k] / total
        results[f"NDCG@{k}"] = ndcgs[k] / total
        results[f"Precision@{k}"] = precision[k] / total
        results[f"Recall@{k}"] = recall[k] / total
    results["Accuracy"] = accuracy / total

    return results


model = NGCF(num_users, num_items, EMB_DIM, N_LAYERS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_loader = DataLoader(
    BPRDataset(train_interactions, train_user_items, num_items),
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("\nTraining NGCF")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for u, p, n in train_loader:
        u, p, n = u.to(device), p.to(device), n.to(device)

        ue, ie = model(norm_adj)
        loss = bpr_loss(ue[u], ie[p], ie[n])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1:02d} | Loss {total_loss/len(train_loader):.4f}")
print(evaluate_full_ranking(model, test_data, norm_adj, num_items, ks=TOPK))
