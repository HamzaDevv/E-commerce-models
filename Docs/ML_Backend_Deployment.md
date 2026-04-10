# ML Backend — Deployment Reference Guide

> Written: April 2026 | Based on pre-deployment research and code optimizations done in this project.
> Start a fresh conversation referencing this file for the actual deployment steps.

---

## 1. What We're Deploying

The `ml_backend/` FastAPI service exposes three AI engines:

| Engine | Technology | Purpose |
|---|---|---|
| **Hybrid Search** | BM25 + Trigram Fuzzy (RapidFuzz) | Product search for 50K catalog |
| **BasketGPT** | PyTorch Transformer (~3.3M params) | Autoregressive cart completion |
| **Basket-RAG** | PyTorch + FAISS IVF-PQ | Contrastive basket similarity recommendations |

---

## 2. Current State of Artifacts (Post-Optimization)

### Search Engine (`data/`)
| File | Size | Notes |
|---|---|---|
| `bm25_index.pkl` | 2.8 MB | BM25 term index |
| `ngram_index.pkl` | 2.5 MB | Trigram fuzzy index |
| `id_mapping.pkl` | 145 KB | Index position → product_id |
| `metadata_map.pkl` | 5.8 MB | product_id → name/price/image |
| ~~`faiss_index.bin`~~ | ~~146 MB~~ | **DELETED** — semantic layer removed |

**No external embedding API needed.** Search is 100% offline (BM25 + fuzzy).

### Basket-RAG Engine (`data/basket_rag/`)
| File | Size | Notes |
|---|---|---|
| `basket_index.faiss` | 69 MB | IVF-PQ compressed (was 1.47 GB `.npy`) |
| `basket_metadata_slim.pkl` | 98 MB | Slimmed (was 134 MB with `user_id` stripped) |
| `encoder_best.pt` | 26 MB | Trained BasketEncoder weights |
| `vocab.pkl` | 2.3 MB | Product token vocab |
| `basket_rag_config.json` | 131 B | Model hyperparams |
| ~~`basket_vectors.npy`~~ | ~~1.47 GB~~ | **Can be deleted** — replaced by `basket_index.faiss` |
| ~~`basket_metadata.pkl`~~ | ~~134 MB~~ | **Can be deleted** — replaced by `basket_metadata_slim.pkl` |

> [!IMPORTANT]
> Before deploying, manually delete `basket_vectors.npy` (1.47 GB) and `basket_metadata.pkl` (134 MB). These are the original files that have been replaced by the compressed versions. They will break your deployment size budget if left in.
> ```bash
> rm ml_backend/data/basket_rag/basket_vectors.npy
> rm ml_backend/data/basket_rag/basket_metadata.pkl
> ```

---

## 3. Measured RAM Usage (All Engines Loaded)

| Engine | RAM Added | Cumulative |
|---|---|---|
| Python + FastAPI baseline | — | ~18 MB |
| BasketGPT (PyTorch) | +214 MB | ~232 MB |
| Basket-RAG (IVF-PQ + metadata) | +1,616 MB | ~1,848 MB |
| Hybrid Search (BM25 + fuzzy) | +85 MB | ~1,934 MB |
| **TOTAL PEAK** | | **~2 GB** |

> [!WARNING]
> The Basket-RAG engine uses ~1.6 GB RAM at runtime even with the compressed FAISS index, because Python holds 3 million `list` entries for basket metadata in-memory. This is the hard floor — **you need at least 2.5 GB RAM headroom** on your host.

---

## 4. Deployment Platform Decision

### ❌ Render (Free Tier) — NOT viable
- **512 MB RAM** hard limit → immediate OOM crash
- No persistent disk on free tier
- 15-minute spin-down causes cold starts

### ❌ Railway (Free Tier) — NOT viable
- ~1 GB RAM limit → still OOM crash
- $5/month hobby plan gives 8 GB but adds cost

### ❌ Qdrant Cloud (Free Tier) — NOT viable for basket vectors
- Only 1 GB RAM, 4 GB disk
- 3M vectors × 128d would exceed even the disk limit

### ✅ Hugging Face Spaces (Free, Docker) — **RECOMMENDED**
- **16 GB RAM**, 2 vCPU — comfortably fits our 2 GB need
- Free tier (CPU basic hardware)
- Deploy via Docker (`Dockerfile` + push to HF repo)
- Designed for ML backends — no spin-down issues
- Persistent within deployment (no persistent volume needed for static artifacts)
- **URL format**: `https://huggingface.co/spaces/<username>/<space-name>`

### ✅ Railway ($5/month hobby) — Viable paid option
- 8 GB RAM, easy GitHub integration
- Good DX, simple deployments

---

## 5. Recommended Deployment: Hugging Face Spaces (Docker)

### 5.1 What You'll Need
1. A **Hugging Face account**: https://huggingface.co
2. A new **Space** created with Docker SDK
3. Git LFS enabled on the repo (for large files)

### 5.2 Files Needed in the HF Space Repo

```
ml_backend/
├── Dockerfile              ← Create this (see section 5.3)
├── main.py
├── requirements.txt
├── routers/
├── models/
├── data/
│   ├── bm25_index.pkl
│   ├── ngram_index.pkl
│   ├── id_mapping.pkl
│   ├── metadata_map.pkl
│   └── basket_rag/
│       ├── basket_index.faiss   (69 MB — needs Git LFS)
│       ├── basket_metadata_slim.pkl  (98 MB — needs Git LFS)
│       ├── encoder_best.pt      (26 MB — needs Git LFS)
│       ├── vocab.pkl
│       └── basket_rag_config.json
```

### 5.3 Dockerfile to Create

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set OpenMP env vars to prevent thread collisions (critical for PyTorch + FAISS)
ENV KMP_DUPLICATE_LIB_OK=True
ENV OMP_NUM_THREADS=1

# HuggingFace Spaces exposes port 7860
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 5.4 Git LFS Setup (Required for large files)

```bash
# Install git-lfs if not already (macOS)
brew install git-lfs
git lfs install

# In your HF Space repo, track large file types
git lfs track "*.faiss"
git lfs track "*.pkl"
git lfs track "*.pt"
git lfs track "*.npy"   # if any remain
git add .gitattributes
```

### 5.5 CORS Update for Production

In `main.py`, update CORS to only allow your Render frontend domain:
```python
allow_origins=["https://your-instamart.onrender.com"]  # Replace with actual Render URL
```

### 5.6 Environment Variables on HF Spaces
The `.env` file should NOT be committed. Set secrets via:
- HF Space UI → Settings → Repository Secrets
- Current keys needed: `GEMINI_API_KEY` (unused now but in .env), `MXBAI_API_KEY`, `CO_API_KEY`
- Actually after removing semantic search, **no API keys are needed at runtime** for search or recommendations. The `.env` is now unused by the running app.

---

## 6. `requirements.txt` (Final Clean Version)

```
fastapi==0.135.3
uvicorn==0.43.0
pandas==3.0.2
rank-bm25==0.2.2
faiss-cpu==1.13.2
scikit-learn==1.8.0
rapidfuzz==3.14.3
python-dotenv==1.2.2
torch==2.11.0
```

No external embedding API dependencies. Fully self-contained.

---

## 7. Search Engine Architecture (Final)

The semantic/vector layer was **removed**. Current engine is:

```
Query → normalize_text()
       ├── Layer 1: Trigram n-gram index → RapidFuzz token_sort_ratio → fuzzy_dict
       └── Layer 2: BM25 → bm25_dict

Combined Score = 0.6 × fuzzy + 0.4 × BM25
              + 0.15 bonus (exact token overlap)
              + 0.05 bonus (prefix match)
```

**Benefits of removing semantic layer:**
- No FAISS index for search (146 MB freed)
- No embedding API calls (no latency, no rate limits, no API key)
- Boots instantly, works fully offline
- Search quality is still excellent for product name matching

---

## 8. Basket-RAG Engine Changes

### What Changed
- `basket_vectors.npy` (1.47 GB) → `basket_index.faiss` (69 MB) using **FAISS IVF-PQ**
  - 96 sub-quantizers × 8 bits → 16 bytes per vector (vs 512 bytes)
  - `nlist=1024, nprobe=32` for search quality
  - ~21× compression, ~5% accuracy loss (acceptable for recommendations)
- `basket_metadata.pkl` (134 MB) → `basket_metadata_slim.pkl` (98 MB)
  - Stripped `user_id` field — only `product_ids` list kept per basket
  - Metadata is now a `list[list[int]]` instead of `list[dict]`
- MMR diversity filter **removed** (not compatible with PQ-approximated vectors)
  - IVF clustering provides implicit diversity
  - Popularity penalty still applied in scoring

### How to Generate Compressed Artifacts (if re-training)
```bash
cd ml_backend
source venv/bin/activate
python3 scripts/compress_basket_index.py   # 1.47 GB → 69 MB (~5-10 min)
python3 scripts/slim_metadata.py           # 134 MB → 98 MB (~instant)
```

---

## 9. Instamart Frontend (Already on Render)

- Deployed on **Render** (free tier — client static site)
- Node.js API server also on Render
- After deploying ML backend on HF Spaces, update the API URL in:
  - `instamart/client/src/` — wherever `ML_API_URL` or `http://localhost:8000` is hardcoded
  - `instamart/server/.env` — `ML_BACKEND_URL` variable (if present)

---

## 10. Deployment Checklist for Next Conversation

- [ ] Delete `basket_vectors.npy` and `basket_metadata.pkl` from the repo
- [ ] Create a HuggingFace account / Space (Docker SDK, Public)
- [ ] Install git-lfs, set up `.gitattributes` for `*.faiss`, `*.pkl`, `*.pt`
- [ ] Write `Dockerfile` (template in section 5.3 above)
- [ ] Push code + artifacts to HF Space repo
- [ ] Verify build logs — watch for OOM or missing file errors
- [ ] Test `/api/search?q=apple` and `/api/recommendation/basket-rag` endpoints
- [ ] Update `allow_origins` in `main.py` with actual Render frontend URL
- [ ] Update frontend API URL pointing to HF Spaces URL
