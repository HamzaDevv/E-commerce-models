# 🛠️ Manual Model Setup Guide

Because the model weights and vector indices are large, they are excluded from the git repository via `.gitignore`. If you are setting up the project from scratch, you must manually place the following artifacts in the `ml_backend/data/` directory.

## 📂 Directory Structure

Ensure your `ml_backend/data/` folder looks like this:

```text
ml_backend/data/
├── basket/                         # BasketGPT (Autoregressive) Artifacts
│   ├── basket_gpt.pt               # Trained PyTorch weights
│   ├── basket_gpt_config.json      # Model hyperparameters
│   └── product_lookup.pkl          # ID ↔ Name mapping for GPT
├── basket_rag/                     # Basket-RAG (Retrieval) Artifacts
│   ├── encoder_best.pt             # Contrastive Encoder weights
│   ├── basket_rag_config.json      # Model hyperparameters
│   ├── basket_vectors.npy          # 3M pre-computed embeddings (1.5GB)
│   ├── basket_metadata.pkl         # Retrieved basket contents
│   └── vocab.pkl                   # Token mappings for RAG
├── bm25_index.pkl                  # Search: Keyword index
├── faiss_index.bin                 # Search: Vector index
├── id_mapping.pkl                  # Search: Index ↔ Product ID map
└── metadata_map.pkl                # Shared product metadata (prices, aisles)
```

---

## 🔍 Search Engine Artifacts
*Place these in the root of `ml_backend/data/`*

- **`bm25_index.pkl`**: Contains the BM25 model for keyword-based relevance.
- **`faiss_index.bin`**: The semantic vector store (Approximate Nearest Neighbors).
- **`id_mapping.pkl`**: Bridges the gap between Faiss/BM25 row indices and actual Product IDs.

## 🏀 BasketGPT Artifacts
*Place these in `ml_backend/data/basket/`*

- **`basket_gpt.pt`**: The transformer weights used for predicting next items in a sequence.
- **`basket_gpt_config.json`**: Defines the `vocab_size`, `n_embd`, `n_head`, and `n_layer`.

## 🔎 Basket-RAG Artifacts
*Place these in `ml_backend/data/basket_rag/`*

- **`encoder_best.pt`**: The contrastive encoder that turns a cart into a retrieval vector.
- **`basket_vectors.npy`**: The massive corpus of 3 million historical shopping trips.
- **`basket_metadata.pkl`**: A lookup table that tells the system exactly which products were in the historical trips found by Faiss.

---

> [!NOTE]
> If any of these files are missing, the corresponding engine will fail to boot during the FastAPI `lifespan` sequence. Check your server logs for "⚠️ Engine failed to load" messages.
