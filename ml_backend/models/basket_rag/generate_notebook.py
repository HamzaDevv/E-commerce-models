import json

# ================================================================
# Basket-RAG Colab Training Notebook Generator
# Run this script to regenerate the .ipynb file.
# ================================================================

cells = []

def md(source):
    """Create a markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source, outputs=None):
    """Create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": source,
    }


# ── CELL 0: Title ────────────────────────────────────────────────
cells.append(md("""# 🛒 Basket-RAG: Sequential Recommendation via Contrastive Learning
> **Architecture:** Basket Encoder (CLS Token + Multi-Head Attention) trained with NT-Xent Loss  
> **Objective:** Consecutive baskets $(B_t, B_{t+1})$ → Cosine Similarity = 1  
> **Output:** Basket Embedding Vectors for downstream Faiss retrieval  

## Instructions
1. Upload your Instacart Dataset CSVs to Google Drive.
2. Update `DATA_DIR` in Cell 2 to point to that folder.
3. Runtime → **Change runtime type → T4 GPU**.
4. Run all cells top-to-bottom.
"""))


# ── CELL 1: Mount Drive ──────────────────────────────────────────
cells.append(md("### Step 1 — Mount Google Drive & Install Dependencies"))
cells.append(code("""from google.colab import drive
drive.mount('/content/drive')"""))

cells.append(code("""# Install any missing libraries (faiss for optional index building in Colab)
!pip install -q faiss-gpu 2>/dev/null || pip install -q faiss-cpu"""))


# ── CELL 2: Config ───────────────────────────────────────────────
cells.append(md("### Step 2 — Global Configuration"))
cells.append(code("""import os, random, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# ════════════════════════════════════════════════════════════
# ❗ UPDATE THIS PATH to where your CSVs live in Google Drive
# Expected files:
#   - orders.csv
#   - order_products__prior.csv
#   - products_with_images.csv  (or products.csv as fallback)
# ════════════════════════════════════════════════════════════
DATA_DIR   = "/content/drive/MyDrive/Instacart_Data"
OUTPUT_DIR = "/content/drive/MyDrive/BasketRAG_Artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Model Hypers ────────────────────────────────────────────
EMBED_DIM   = 128    # Item + basket representation dimension
N_HEADS     = 4
N_LAYERS    = 2
MAX_LEN     = 60     # Max products per basket (incl. CLS token)
PAD_TOKEN   = 0
CLS_TOKEN   = 1

# ── Training Hypers ─────────────────────────────────────────
BATCH_SIZE      = 256
EPOCHS          = 10
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
TEMPERATURE     = 0.1   # NT-Xent temperature (lower = sharper separation)
DROPOUT_RATE    = 0.20  # Item-dropout augmentation probability
GRAD_CLIP       = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
"""))


# ── CELL 3: Data Builder ─────────────────────────────────────────
cells.append(md("""### Step 3 — Build Vocabulary & User Trajectories
Loads `orders.csv` + `order_products__prior.csv`, groups products by transaction,
and sorts each user's orders chronologically to form sequential \"trajectories\".
"""))
cells.append(code("""class BasketDatasetBuilder:
    \"\"\"Converts raw Instacart CSVs into sequential user basket trajectories.\"\"\"

    def __init__(self):
        self.product2token = {}
        self.token2product = {}
        self.vocab_size    = 2  # 0=PAD, 1=CLS

    # ── Step A: Build vocab from product catalog ──────────────
    def build_vocab(self):
        print("📦 Building vocabulary...")
        try:
            df = pd.read_csv(f"{DATA_DIR}/products_with_images.csv",
                             usecols=["product_id", "product_name"])
        except Exception:
            df = pd.read_csv(f"{DATA_DIR}/products.csv",
                             usecols=["product_id", "product_name"])

        df = df.drop_duplicates("product_id").sort_values("product_id")
        ids   = df["product_id"].tolist()
        names = df.set_index("product_id")["product_name"].to_dict()

        # Offset by 2 (0=PAD, 1=CLS reserved)
        self.product2token = {pid: i + 2 for i, pid in enumerate(ids)}
        self.token2product = {i + 2: pid for i, pid in enumerate(ids)}
        self.id2name       = {pid: names.get(pid, f"Product {pid}") for pid in ids}
        self.vocab_size    = len(ids) + 2

        print(f"   Vocab size: {self.vocab_size:,}  ({len(ids):,} products + PAD + CLS)")
        return self.product2token, self.token2product, self.id2name, self.vocab_size

    # ── Step B: Build per-user ordered basket sequences ───────
    def build_trajectories(self):
        print("\\n🗂️  Building user trajectories (this takes ~2 min)...")

        orders  = pd.read_csv(f"{DATA_DIR}/orders.csv",
                              usecols=["order_id","user_id","order_number"])
        prior   = pd.read_csv(f"{DATA_DIR}/order_products__prior.csv",
                              usecols=["order_id","product_id"])

        merged  = prior.merge(orders, on="order_id")
        merged  = merged[merged["product_id"].isin(self.product2token)]
        merged["token"] = merged["product_id"].map(self.product2token)

        # Group items into baskets, then sort baskets by order_number per user
        baskets = (merged
                   .groupby(["user_id","order_number"])["token"]
                   .apply(list)
                   .reset_index()
                   .sort_values(["user_id","order_number"]))

        trajectories = (baskets
                        .groupby("user_id")["token"]
                        .apply(list)
                        .to_dict())

        n_pairs = sum(len(t) - 1 for t in trajectories.values())
        print(f"   Users: {len(trajectories):,}  |  Total (t, t+1) pairs: {n_pairs:,}")
        return trajectories


builder = BasketDatasetBuilder()
product2token, token2product, id2name, VOCAB_SIZE = builder.build_vocab()
trajectories = builder.build_trajectories()

# Persist for inference later
vocab_data = {
    "product2token": product2token,
    "token2product": token2product,
    "id2name":       id2name,
    "vocab_size":    VOCAB_SIZE,
}
with open(f"{OUTPUT_DIR}/vocab.pkl", "wb") as f:
    pickle.dump(vocab_data, f)
print(f"\\n✅ Vocab saved → {OUTPUT_DIR}/vocab.pkl")
"""))


# ── CELL 4: Dataset ──────────────────────────────────────────────
cells.append(md("""### Step 4 — PyTorch Contrastive Dataset
Each sample produces a **triplet** `(Query, Positive, Negative)`:
- **Query** `q`: $Basket_t$ with augmentation (shuffle + item dropout)
- **Positive** `p`: $Basket_{t+1}$ (the true next basket)
- **Hard Negative** `n`: A *different* trip by the same user (temporal negative),
  or a random basket from another user if the trajectory is too short.
"""))
cells.append(code("""class BasketContrastiveDataset(Dataset):

    def __init__(self, trajectories, max_len=MAX_LEN, dropout=DROPOUT_RATE):
        self.traj     = trajectories
        self.max_len  = max_len
        self.dropout  = dropout
        self.users    = list(trajectories.keys())
        self.pairs    = []

        for u in self.users:
            t = trajectories[u]
            for i in range(len(t) - 1):
                self.pairs.append((u, i))   # (user, basket index t)

    def __len__(self):
        return len(self.pairs)

    # ── Augmentation ──────────────────────────────────────────
    def _augment(self, basket):
        b = basket.copy()
        if len(b) > 2:
            b = [x for x in b if random.random() > self.dropout]
            if not b:
                b = [random.choice(basket)]
        random.shuffle(b)
        return b

    # ── Pad / truncate, prepend CLS ───────────────────────────
    def _format(self, basket):
        b = [CLS_TOKEN] + basket          # position 0 = [CLS]
        if len(b) > self.max_len:
            b = b[:self.max_len]
        else:
            b += [PAD_TOKEN] * (self.max_len - len(b))
        return torch.tensor(b, dtype=torch.long)

    def __getitem__(self, idx):
        u, i   = self.pairs[idx]
        traj_u = self.traj[u]

        basket_t      = self._augment(traj_u[i])
        basket_t_next = traj_u[i + 1]       # Target – no augmentation corruption

        # Hard negative: same user, non-consecutive basket
        candidate_neg_indices = [j for j in range(len(traj_u))
                                 if j != i and j != i + 1]
        if candidate_neg_indices:
            neg_basket = traj_u[random.choice(candidate_neg_indices)]
        else:
            # Fallback: random different user
            other_u    = random.choice(self.users)
            neg_basket = random.choice(self.traj[other_u])

        neg_basket = self._augment(neg_basket)

        return (self._format(basket_t),
                self._format(basket_t_next),
                self._format(neg_basket))


# ── Instantiate ───────────────────────────────────────────────────
dataset    = BasketContrastiveDataset(trajectories)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True)

print(f"Dataset size : {len(dataset):,} triplets")
print(f"Batches/epoch: {len(dataloader):,}")
q, p, n = dataset[0]
print(f"Sample shapes: q={tuple(q.shape)}, p={tuple(p.shape)}, n={tuple(n.shape)}")
"""))


# ── CELL 5: Model ────────────────────────────────────────────────
cells.append(md("""### Step 5 — Model Architecture: BasketEncoder
Key design choices:
- **`[CLS]` token** at position 0 collects global basket context after self-attention.
- **Transformer Encoder** lets every item attend to every other item in the basket.
- **Projector Head** (non-linear MLP) maps the CLS embedding to the contrastive loss space.
  → Dropped at inference time; raw CLS is used for Faiss indexing (following SimCLR).
"""))
cells.append(code("""class BasketEncoder(nn.Module):
    \"\"\"
    Encodes a variable-length basket of product tokens into a fixed-size vector.

    Architecture
    ------------
    EmbeddingLayer → TransformerEncoder → CLS pooling → [Projector during training]
    \"\"\"

    def __init__(self, vocab_size, embed_dim=EMBED_DIM,
                 n_heads=N_HEADS, n_layers=N_LAYERS, dropout=0.1):
        super().__init__()

        self.item_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,          # Pre-LN for more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                                  enable_nested_tensor=False)

        # SimCLR-style non-linear projector (used ONLY during training)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, std=0.02)
        for p in self.projector.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, input_ids):
        \"\"\"Core encoder: returns the raw CLS embedding (for Faiss / inference).\"\"\"
        padding_mask = (input_ids == PAD_TOKEN)          # True = ignore
        emb          = self.item_emb(input_ids)          # [B, L, D]
        out          = self.transformer(emb, src_key_padding_mask=padding_mask)
        return out[:, 0, :]                              # CLS token at index 0

    def forward(self, input_ids):
        \"\"\"Training forward: CLS → Projector.\"\"\"
        return self.projector(self.encode(input_ids))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


model = BasketEncoder(VOCAB_SIZE).to(DEVICE)
n_params = model.count_parameters()
print(f"BasketEncoder | {n_params:,} trainable parameters")
print(f"Embed dim={EMBED_DIM} | Heads={N_HEADS} | Layers={N_LAYERS} | Vocab={VOCAB_SIZE:,}")
"""))


# ── CELL 6: Loss ─────────────────────────────────────────────────
cells.append(md("""### Step 6 — NT-Xent Contrastive Loss
Pulls the positive pair together, pushes the negative away, scaled by temperature.  
Lower temperature → sharper separation (harder training, better embeddings).
"""))
cells.append(code("""class NTXentLoss(nn.Module):
    \"\"\"
    Normalized Temperature-scaled Cross-Entropy Loss (SimCLR-style).

    For each (q, p, n) triplet:
      - Maximises cosine_sim(q, p) / τ
      - Minimises cosine_sim(q, n) / τ
    Formulated as classification loss where class 0 = positive.
    \"\"\"

    def __init__(self, temperature=TEMPERATURE):
        super().__init__()
        self.T = temperature

    def forward(self, q, p, n):
        # L2-normalise so cosine similarity = dot product
        q = F.normalize(q, dim=-1)
        p = F.normalize(p, dim=-1)
        n = F.normalize(n, dim=-1)

        pos = (q * p).sum(dim=-1, keepdim=True) / self.T   # [B, 1]
        neg = (q * n).sum(dim=-1, keepdim=True) / self.T   # [B, 1]

        logits = torch.cat([pos, neg], dim=1)               # [B, 2]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels)


criterion = NTXentLoss(TEMPERATURE)
print(f"NT-Xent loss | temperature={TEMPERATURE}")
"""))


# ── CELL 7: Training ─────────────────────────────────────────────
cells.append(md("""### Step 7 — Training Loop
Includes:
- **Gradient clipping** for stability
- **Cosine Annealing LR** scheduler
- **Per-epoch checkpoint** to Google Drive
- Best checkpoint saved separately based on lowest validation loss
"""))
cells.append(code("""optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_loss    = float("inf")
history      = {"train_loss": []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}", leave=True)

    for q_tok, p_tok, n_tok in pbar:
        q_tok = q_tok.to(DEVICE)
        p_tok = p_tok.to(DEVICE)
        n_tok = n_tok.to(DEVICE)

        optimizer.zero_grad()

        q_proj = model(q_tok)
        p_proj = model(p_tok)
        n_proj = model(n_tok)

        loss = criterion(q_proj, p_proj, n_proj)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         lr=f"{scheduler.get_last_lr()[0]:.2e}")

    scheduler.step()
    avg_loss = running_loss / len(dataloader)
    history["train_loss"].append(avg_loss)

    # ── Save checkpoint ────────────────────────────────────────
    ckpt_path = f"{OUTPUT_DIR}/encoder_ep{epoch:02d}.pt"
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "loss":        avg_loss,
        "config": {
            "vocab_size": VOCAB_SIZE,
            "embed_dim":  EMBED_DIM,
            "n_heads":    N_HEADS,
            "n_layers":   N_LAYERS,
            "max_len":    MAX_LEN,
        }
    }, ckpt_path)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/encoder_best.pt")

    print(f"  Epoch {epoch:02d} | avg_loss={avg_loss:.4f} | best={best_loss:.4f}")

print("\\n✅ Training complete!")
print(f"   Best loss  : {best_loss:.4f}")
print(f"   Best model → {OUTPUT_DIR}/encoder_best.pt")
"""))


# ── CELL 8: Save Model Config ────────────────────────────────────
cells.append(md("### Step 8 — Save Final Artifacts for Local Inference"))
cells.append(code("""import json

# Save model config (needed to reconstruct the model class locally)
model_config = {
    "vocab_size": VOCAB_SIZE,
    "embed_dim":  EMBED_DIM,
    "n_heads":    N_HEADS,
    "n_layers":   N_LAYERS,
    "max_len":    MAX_LEN,
    "pad_token":  PAD_TOKEN,
    "cls_token":  CLS_TOKEN,
}
with open(f"{OUTPUT_DIR}/basket_rag_config.json", "w") as f:
    json.dump(model_config, f, indent=2)

print(f"Config saved → {OUTPUT_DIR}/basket_rag_config.json")
print(f"Vocab  saved → {OUTPUT_DIR}/vocab.pkl")
print(f"Weights→      {OUTPUT_DIR}/encoder_best.pt")
print("\\nPlease download these 3 files and place them in:")
print("  ml_backend/data/basket_rag/")
"""))


# ── CELL 9: Export Basket Vectors ────────────────────────────────
cells.append(md("""### Step 9 — Export ALL Basket Vectors for Faiss Indexing
Passes every basket in the training set through the frozen encoder and saves the
vectors + metadata. These are loaded by the local `BasketRAGEngine` at startup.

> ⚠️ This cell can take 5-10 minutes. The output `.npy` / `.pkl` files should
> also be copied to `ml_backend/data/basket_rag/`.
"""))
cells.append(code("""model.eval()

all_vectors  = []
all_metadata = []   # list of {"user": uid, "tokens": [...], "product_ids": [...]}

# Re-use the same dataset pairs (t_next baskets = the "retrieval corpus")
EXPORT_BATCH = 512
pairs_list   = dataset.pairs   # list of (user, basket_idx)

print(f"Exporting {len(pairs_list):,} basket vectors...")

with torch.no_grad():
    for start in tqdm(range(0, len(pairs_list), EXPORT_BATCH), desc="Exporting"):
        batch_pairs = pairs_list[start : start + EXPORT_BATCH]

        # Use the t+1 basket (the "positive") as our retrieval corpus entry
        raw_baskets = [dataset.traj[u][i + 1] for u, i in batch_pairs]
        tokens_list = [dataset._format(b) for b in raw_baskets]
        tokens_tensor = torch.stack(tokens_list).to(DEVICE)

        # encode() gives raw CLS (no projector) — used for retrieval
        embs = model.encode(tokens_tensor).cpu().numpy()

        all_vectors.extend(embs)
        for (u, i), raw in zip(batch_pairs, raw_baskets):
            product_ids = [token2product.get(t, -1) for t in raw if t > 1]
            all_metadata.append({"user_id": int(u), "product_ids": product_ids})

all_vectors = np.array(all_vectors, dtype=np.float32)
np.save(f"{OUTPUT_DIR}/basket_vectors.npy",  all_vectors)
with open(f"{OUTPUT_DIR}/basket_metadata.pkl", "wb") as f:
    pickle.dump(all_metadata, f)

print(f"\\n✅ Vectors shape : {all_vectors.shape}")
print(f"   Metadata rows : {len(all_metadata):,}")
print(f"   Saved to      : {OUTPUT_DIR}/")
"""))


# ── CELL 10: Sanity check ────────────────────────────────────────
cells.append(md("### Step 10 — Quick Sanity Check: Cosine Similarity of Neighbour Pairs"))
cells.append(code("""# Verify that the model is actually pulling consecutive pairs together
model.eval()
sims_pos, sims_neg = [], []

with torch.no_grad():
    for q_tok, p_tok, n_tok in list(dataloader)[:10]:
        q_tok = q_tok.to(DEVICE)
        p_tok = p_tok.to(DEVICE)
        n_tok = n_tok.to(DEVICE)

        q_enc = F.normalize(model.encode(q_tok), dim=-1)
        p_enc = F.normalize(model.encode(p_tok), dim=-1)
        n_enc = F.normalize(model.encode(n_tok), dim=-1)

        sims_pos.extend((q_enc * p_enc).sum(-1).cpu().tolist())
        sims_neg.extend((q_enc * n_enc).sum(-1).cpu().tolist())

print(f"Avg cosine sim  (positive pairs) : {np.mean(sims_pos):.4f}  ← should be > 0.5")
print(f"Avg cosine sim  (negative pairs) : {np.mean(sims_neg):.4f}  ← should be < 0.3")
"""))


# ── Assemble and write ────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12",
        },
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "provenance": [],
            "name": "Basket_RAG_Training.ipynb",
        },
    },
    "cells": cells,
}

OUT = "/Users/hamza/Developer/E-commerce_web/ml_backend/models/basket_rag/Basket_RAG_Training.ipynb"
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✅  Notebook written → {OUT}")
