# Instamart ML Backend API

A high-performance, modular FastAPI backend for e-commerce intelligence. This repository contains advanced machine learning models for search, product recommendations, and basket analysis.

## 🚀 Features

### 1. Hybrid E-Commerce Search
- **BM25 Keyword Search**: Robust keyword matching for relevant product discovery.
- **FAISS Semantic Similarity**: Vector-based search using `nomic-embed-text` embeddings from Ollama.
- **Typo Tolerance**: Hybrid scoring mechanism to handle search inaccuracies.

### 2. Basket-RAG Recommendation Engine
- **Retrieval-Augmented Generation**: Treats shopping as a retrieval problem across a 3-million-basket historical corpus.
- **Contrastive Learning**: Uses a Transformer-based `BasketEncoder` trained with NT-Xent loss to map shopping intent to dense vectors.
- **Maximal Marginal Relevance (MMR)**: Ensures diversity in retrieved candidates to prevent redundant recommendations.
- **Faron's F1 Optimization**: Dynamically determines the optimal number of items to recommend for maximum precision/recall.

### 3. BasketGPT Completion Engine
- **Autoregressive Transformer**: A lightweight PyTorch-based model for predicting next-item probability sequences.
- **RoPE Embeddings**: Utilizes Rotary Position Embeddings for high-quality sequence modeling.

### 4. Modular API Architecture
- Clean FastAPI routing (`/search`, `/recommendations`, `/basket-rag`).
- Environment-based configuration using `.env`.
- Sequential "Three-Engine Boot" process to prevent OpenMP thread collisions.
- Scalable backend for processing large product catalogs (up to 50k+ items).

## 🛠 Tech Stack
- **Framework**: FastAPI, Uvicorn
- **Search Logic**: Rank-BM25, FAISS (CPU)
- **ML/DL**: PyTorch, Transformers, Scikit-learn
- **Embeddings**: Ollama (nomic-embed-text)
- **Data Handling**: Pandas

## 📦 Setup & Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/HamzaDevv/E-commerce-models.git
   cd E-commerce-models
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r ml_backend/requirements.txt
   ```

4. **Environment Configuration**:
   Create a `.env` file in the `ml_backend/` directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   ```

5. **Run the Server (Development)**:
   ```bash
   cd ml_backend
   uvicorn main:app --reload --port 8000
   ```

6. **Run the Server (Production)**:
   The application uses a strict sequential `lifespan` initialization to prevent OpenMP crashes (PyTorch + Faiss collisions) and manages CPU-bound inference efficiently.
   ```bash
   cd ml_backend
   uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000
   ```
   *Note: The boot sequence follows: BasketGPT → Basket-RAG → Hybrid Search.*

## 📂 Project Structure
```text
.
├── ml_backend/
│   ├── main.py             # Entry point with sequential engine boot
│   ├── routers/            # Search, Recommendation, and Basket-RAG routers
│   ├── models/             # Transformer (GPT) and Contrastive (RAG) architectures
│   ├── data/               # (Excluded) Model weights and Vector Indices
│   └── requirements.txt    # Python dependencies (pinned)
├── Dataset/                # (Excluded) Raw CSV datasets
├── Docs/                   # Detailed Model Architectures & Journey
│   └── Model_readme/       # READMEs for BasketGPT and Basket-RAG
└── README.md
```

## ⚠️ Important Notes
- Large datasets (`Dataset/`) and generated indices (`ml_backend/data/`) are excluded via `.gitignore` to maintain repository speed and size.
- Ensure Ollama is running locally if using semantic search features.
