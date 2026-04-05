# 🚀 API Integration & AI Agent Guide

This document defines the REST API endpoints provided by the Instamart ML Backend. It is designed for both human developers and AI coding agents to integrate intelligence features into the e-commerce storefront.

---

## 🏗️ Base Configuration
- **Default Base URL**: `http://localhost:8000`
- **Authentication**: None (handled via CORS whitelist in `main.py`).

---

## 🔍 1. Hybrid Search Engine
Perform high-speed hybrid (keyword + semantic) search across the product catalog.

- **Endpoint**: `POST /api/search/`
- **Description**: Combines BM25 and Faiss Vector search for typo-tolerant, meaning-aware results.

### Request Body (`JSON`)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `query` | `string` | **Required** | The user's search text. |
| `top_k` | `integer` | `10` | Number of results to return. |

### Example Response
```json
{
  "query": "organic apples",
  "results": [
    { "product_id": 13176, "product_name": "Bag of Organic Bananas", "score": 24.5 },
    { "product_id": 21137, "product_name": "Organic Strawberries", "score": 22.1 }
  ]
}
```

---

## 🛒 2. Basket-RAG Recommendations
Generate "Collaborative" recommendations by retrieving similar historical shopping trips.

- **Endpoint**: `POST /basket-rag/`
- **Description**: Uses Retrieval-Augmented Generation principles to find what other people bought with similar carts.

### Request Body (`JSON`)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `cart_ids` | `array[int]` | **Required** | List of Product IDs currently in the user's cart. |
| `num_suggestions` | `integer` | `5` | How many products to recommend. |
| `use_mmr` | `boolean` | `true` | Whether to use diversity filtering (Maximal Marginal Relevance). |
| `diversity` | `float` | `0.5` | Diversity threshold (0.0 to 1.0) for MMR. |

### Example Response
```json
{
  "cart_context": ["Organic Milk", "Eggs"],
  "recommendations": [
    { "id": 5432, "name": "Organic Whole Wheat Bread", "score": 14.2 },
    { "id": 8901, "name": "Unsalted Butter", "score": 9.5 }
  ],
  "latency_ms": 38.5
}
```

---

## 🔮 3. BasketGPT Completion
Predict the *next* most likely item in a sequence (syntax-aware).

- **Endpoint**: `POST /recommendations/completions`
- **Description**: An autoregressive transformer (BasketGPT) that treats shopping as a language modeling problem.

### Request Body (`JSON`)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `cart_product_ids` | `array[int]` | **Required** | Sequence of Product IDs in the order they were added. |
| `num_suggestions` | `integer` | `5` | Number of items to predict. |
| `temperature` | `float` | `0.7` | Sampling randomness (higher = more creative). |
| `top_k` | `integer` | `20` | Top-K sampling filter. |

---

## 🩺 4. System Health Checks
Check if specific ML engines are initialized.

| Endpoint | Engine |
| :--- | :--- |
| `GET /api/search/health` | Hybrid Search |
| `GET /basket-rag/health` | Basket-RAG |
| `GET /recommendations/health` | BasketGPT |

---

## 🤖 AI Agent Implementation Logic

If you are an AI agent building the frontend logic, follow this strategy:

1.  **Search First**: When a user type in the search bar, use `/api/search/` with `top_k=20`.
2.  **Hybrid Recommendations**:
    - For the **"Customers also bought..."** section, primary call should be `/basket-rag/`.
    - For a **"Complete your meal"** popup, use `/recommendations/completions` as it focuses on sequence prediction.
3.  **Boot Handling**: If an endpoint returns a `500` status with a "not loaded" message, notify the user that the background model is still initializing (it takes ~8 seconds for a cold start).
