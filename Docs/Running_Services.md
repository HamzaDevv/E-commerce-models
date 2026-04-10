# Running the Instamart Services

This reference details the directories, commands, and expected local URLs for starting and developing the complete Instamart application.

## 1. ML Backend (FastAPI)
The python-based backbone for advanced machine learning models including Hybrid Search, BasketGPT, and Basket-RAG.

- **Working Directory**: `ml_backend/`
- **Startup Command**:
  ```bash
  source venv/bin/activate && uvicorn main:app --reload --port 8000
  ```
- **Local URL**: `http://127.0.0.1:8000`
- **Notes**: On startup, it performs a strict sequential "Three-Engine Boot" (BasketGPT -> Basket-RAG -> Hybrid Search) to safely load the large PyTorch and Faiss models into memory without OpenMP thread collisions.

## 2. Node.js API (Express)
The main API server for user authentication, managing cart items, checking out, and product database interactions.

- **Working Directory**: `instamart/server/`
- **Startup Command**:
  ```bash
  node server.js
  ```
  *(You can also use `npx nodemon server.js` if you want auto-reloading during development.)*
- **Local URL**: `http://localhost:5001` (Port 5001)

## 3. Instamart Frontend (React/Vite)
The user interface where shoppers interact with the platform, search for items, and see RAG-powered cart recommendations.

- **Working Directory**: `instamart/client/`
- **Startup Command**:
  ```bash
  npm run start
  ```
- **Local URL**: `http://localhost:5173/`

## Running all 3 Simultaneously in Background (Agent Reference)
If you are asking the AI agent to start everything, it will typically background these processes using persistent terminal sessions in their respective directories context.
