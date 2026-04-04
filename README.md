# Instamart ML Backend API

A high-performance, modular FastAPI backend for e-commerce intelligence. This repository contains advanced machine learning models for search, product recommendations, and basket analysis.

## 🚀 Features

### 1. Hybrid E-Commerce Search
- **BM25 Keyword Search**: Robust keyword matching for relevant product discovery.
- **FAISS Semantic Similarity**: Vector-based search using `nomic-embed-text` embeddings from Ollama.
- **Typo Tolerance**: Hybrid scoring mechanism to handle search inaccuracies.

### 2. BasketGPT Recommendation Engine
- **Autoregressive Transformer**: A lightweight PyTorch-based transformer model (BasketGPT) for basket completion.
- **RoPE Embeddings**: Utilizes Rotary Position Embeddings for high-quality sequence modeling.
- **Dynamic Predictor**: Predicts the next most likely item based on current basket contents.

### 3. Modular API Architecture
- Clean FastAPI routing (`/search`, `/recommendations`).
- Environment-based configuration using `.env`.
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
   The application uses a strict sequential `lifespan` initialization to prevent OpenMP crashes and manages CPU-bound inference in an external threadpool. It is safe to run with multiple workers.
   ```bash
   cd ml_backend
   uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000
   ```
   *Note: CORS is enabled by default for all origins to easily connect with the Frontend.*

## 📂 Project Structure
```text
.
├── ml_backend/
│   ├── main.py             # Entry point
│   ├── routers/            # Search and Recommendation routers
│   ├── models/             # Model architectures and logic
│   ├── data/               # (Excluded) FAISS/BM25 Indices
│   └── requirements.txt    # Python dependencies
├── Dataset/                # (Excluded) Raw CSV datasets
└── Docs/                   # Project documentation
```

## ⚠️ Important Notes
- Large datasets (`Dataset/`) and generated indices (`ml_backend/data/`) are excluded via `.gitignore` to maintain repository speed and size.
- Ensure Ollama is running locally if using semantic search features.
