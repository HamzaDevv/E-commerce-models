"""
BasketRAGEngine — Runtime inference engine for the Basket-RAG recommendation system.

Pipeline:
    1. Encode the user's current cart → query vector (CLS embedding)
    2. Search Faiss index for Top-K most similar historical basket vectors
    3. Extract product IDs from those baskets, score by similarity weight
    4. Apply recency/popularity bias adjustments
    5. Return ranked recommendations (with optional F1-optimised cutoff)

Artifacts required (copy from Colab OUTPUT_DIR):
    ml_backend/data/basket_rag/
        ├── encoder_best.pt          — trained encoder weights
        ├── basket_rag_config.json   — model hyperparams
        ├── vocab.pkl                — product2token / token2product / id2name
        ├── basket_vectors.npy       — all exported basket embeddings
        └── basket_metadata.pkl      — per-vector product_ids + user_id
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict
from typing import List, Dict, Optional


# ── Model definition (must match training) ──────────────────────────────────────

class BasketEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, n_heads=4, n_layers=2, pad_token=0):
        super().__init__()
        self.pad_token = pad_token
        self.item_emb  = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,          # No dropout during inference
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                                  enable_nested_tensor=False)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns raw CLS embedding (no projector). Used for Faiss and inference."""
        padding_mask = (input_ids == self.pad_token)
        emb = self.item_emb(input_ids)
        out = self.transformer(emb, src_key_padding_mask=padding_mask)
        return out[:, 0, :]   # CLS token at index 0

    def forward(self, input_ids):
        return self.projector(self.encode(input_ids))


# ── Faiss import (lazy, to respect PyTorch-first lifespan ordering) ─────────────

def _import_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        raise RuntimeError(
            "faiss-cpu is not installed. "
            "Run: pip install faiss-cpu"
        )


# ── Main engine ─────────────────────────────────────────────────────────────────

class BasketRAGEngine:
    """
    Retrieval-Augmented Recommendation engine based on contrastive basket embeddings.

    Usage:
        engine = BasketRAGEngine()
        result = engine.recommend(cart_product_ids=[24852, 13176, 21137], top_k=10)
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Resolve relative to this file: ml_backend/models/basket_rag/engine.py
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(base, "data", "basket_rag")
        self.data_dir = data_dir
        self._load_all()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _load_all(self):
        self._check_artifacts()
        self._load_vocab()
        self._load_model()
        self._build_faiss_index()
        self._compute_global_popularity()
        print("✅ BasketRAGEngine ready.")

    def _check_artifacts(self):
        required = [
            "encoder_best.pt",
            "basket_rag_config.json",
            "vocab.pkl",
            "basket_vectors.npy",
            "basket_metadata.pkl",
        ]
        missing = [f for f in required if not os.path.exists(os.path.join(self.data_dir, f))]
        if missing:
            raise FileNotFoundError(
                f"BasketRAG artifacts missing from '{self.data_dir}': {missing}\n"
                "Train the model in Colab and copy the output files."
            )

    def _load_vocab(self):
        print("  [1/4] Loading vocabulary...")
        with open(os.path.join(self.data_dir, "vocab.pkl"), "rb") as f:
            vocab = pickle.load(f)
        self.product2token: Dict[int, int] = vocab["product2token"]
        self.token2product: Dict[int, int] = vocab["token2product"]
        self.id2name:       Dict[int, str] = vocab.get("id2name", {})
        self.vocab_size:    int            = vocab["vocab_size"]
        self.PAD_TOKEN = 0
        self.CLS_TOKEN = 1
        print(f"     Vocab: {self.vocab_size:,} tokens")

    def _load_model(self):
        print("  [2/4] Loading BasketEncoder weights...")
        config_path  = os.path.join(self.data_dir, "basket_rag_config.json")
        weights_path = os.path.join(self.data_dir, "encoder_best.pt")

        with open(config_path) as f:
            cfg = json.load(f)

        self.config  = cfg
        self.max_len = cfg.get("max_len", 60)

        self.model = BasketEncoder(
            vocab_size=cfg["vocab_size"],
            embed_dim=cfg["embed_dim"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            pad_token=cfg.get("pad_token", 0),
        )

        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        # Support both raw state_dict and checkpoint dicts
        if "model_state" in state:
            state = state["model_state"]
        self.model.load_state_dict(state)
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"     Params: {n_params:,}  |  embed_dim={cfg['embed_dim']}  "
              f"|  layers={cfg['n_layers']}  |  heads={cfg['n_heads']}")

    def _build_faiss_index(self):
        print("  [3/4] Building Faiss index...")
        faiss = _import_faiss()

        vectors_path  = os.path.join(self.data_dir, "basket_vectors.npy")
        metadata_path = os.path.join(self.data_dir, "basket_metadata.pkl")

        self.basket_vectors  = np.load(vectors_path).astype(np.float32)
        with open(metadata_path, "rb") as f:
            self.basket_metadata = pickle.load(f)

        # L2-normalise so cosine similarity = inner product
        norms = np.linalg.norm(self.basket_vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self.basket_vectors_normed = (self.basket_vectors / norms).astype(np.float32)

        d = self.basket_vectors_normed.shape[1]
        self.faiss_index = faiss.IndexFlatIP(d)        # Inner-product = cosine after L2-norm
        self.faiss_index.add(self.basket_vectors_normed)

        print(f"     Index: {self.faiss_index.ntotal:,} vectors  |  dim={d}")

    def _compute_global_popularity(self):
        """Pre-compute global product frequency across all indexed baskets."""
        print("  [4/4] Computing global product popularity...")
        freq: Counter = Counter()
        for meta in self.basket_metadata:
            freq.update(meta["product_ids"])
        total = max(sum(freq.values()), 1)
        self.global_popularity: Dict[int, float] = {
            pid: cnt / total for pid, cnt in freq.items()
        }
        print(f"     Tracked {len(freq):,} unique products in corpus.")

    # ── Encoding helpers ─────────────────────────────────────────────────────

    def _format_basket(self, product_ids: List[int]) -> torch.Tensor:
        """Convert a list of product_ids to a padded token tensor."""
        tokens = [self.product2token[pid] for pid in product_ids
                  if pid in self.product2token]
        if not tokens:
            tokens = [self.CLS_TOKEN]   # fallback
        seq = [self.CLS_TOKEN] + tokens
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        else:
            seq += [self.PAD_TOKEN] * (self.max_len - len(seq))
        return torch.tensor(seq, dtype=torch.long).unsqueeze(0)   # [1, L]

    def _encode_cart(self, product_ids: List[int]) -> np.ndarray:
        """Encode cart → normalised float32 query vector."""
        with torch.no_grad():
            token_tensor = self._format_basket(product_ids)
            emb = self.model.encode(token_tensor)           # [1, D]
            emb = F.normalize(emb, dim=-1)
        return emb.squeeze(0).numpy().astype(np.float32)

    # ── Retrieval ────────────────────────────────────────────────────────────

    def _retrieve(self, query_vec: np.ndarray, n_retrieve: int = 100):
        """
        ANN search with Maximal Marginal Relevance (MMR) diversity filter.
        Returns list of (similarity_score, basket_metadata_dict).
        """
        # Fetch 3× candidates before MMR pruning
        fetch_k = min(n_retrieve * 3, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(query_vec[np.newaxis], fetch_k)
        scores  = scores[0]
        indices = indices[0]

        # MMR: iteratively pick the candidate that max(sim_to_query - sim_to_selected)
        selected_vecs  = []
        selected_pairs = []
        lambda_mmr     = 0.6   # 0 = pure diversity, 1 = pure similarity

        for idx, score in zip(indices, scores):
            if idx < 0:
                continue
            candidate_vec = self.basket_vectors_normed[idx]

            if not selected_vecs:
                selected_vecs.append(candidate_vec)
                selected_pairs.append((float(score), self.basket_metadata[idx]))
            else:
                sim_to_selected = max(
                    float(np.dot(candidate_vec, sv)) for sv in selected_vecs
                )
                mmr_score = lambda_mmr * float(score) - (1 - lambda_mmr) * sim_to_selected
                if mmr_score > 0:
                    selected_vecs.append(candidate_vec)
                    selected_pairs.append((float(score), self.basket_metadata[idx]))

            if len(selected_pairs) >= n_retrieve:
                break

        return selected_pairs

    # ── Scoring ──────────────────────────────────────────────────────────────

    def _score_candidates(
        self,
        retrieved: list,
        cart_ids: List[int],
        recency_boost: float = 0.05,
        popularity_penalty: float = 0.3,
    ) -> Dict[int, float]:
        """
        Score each candidate product:
            score(item) = Σ sim(q, B) × I(item ∈ B)
                        + recency_boost × global_popularity
                        - popularity_penalty × global_popularity  (prevents top-sellers bias)
        """
        cart_set = set(cart_ids)
        scores: defaultdict = defaultdict(float)
        frequency: Counter  = Counter()

        for sim_score, meta in retrieved:
            for pid in meta["product_ids"]:
                if pid in cart_set:
                    continue   # Don't recommend what's already in cart
                scores[pid]    += sim_score
                frequency[pid] += 1

        # Apply popularity adjustment: slight boost but heavy penalty for over-popular items
        for pid in list(scores.keys()):
            pop = self.global_popularity.get(pid, 0.0)
            scores[pid] += recency_boost * pop
            scores[pid] -= popularity_penalty * pop   # Net: discourages "Apple/Avocado" dominance

        return dict(scores)

    # ── F1-Optimal Cutoff (Faron's Algorithm) ────────────────────────────────

    @staticmethod
    def _faron_f1_cutoff(scores: Dict[int, float], min_k: int = 3) -> int:
        """
        Determine the optimal number of recommendations K that maximises
        expected F1-score via Faron's EM method.

        Raw scores are first normalised to [0, 1] probabilities using
        min-max scaling so Faron's multiplicative model works correctly.

        Returns K (number of items to recommend), with a minimum of min_k.
        """
        sorted_scores = sorted(scores.values(), reverse=True)
        if not sorted_scores:
            return min_k

        # Normalise to probabilities via min-max scaling
        max_s = sorted_scores[0]
        min_s = sorted_scores[-1] if len(sorted_scores) > 1 else 0.0
        spread = max_s - min_s if max_s != min_s else 1.0

        best_f1, best_k = 0.0, min_k
        p_none = 1.0  # probability user wants nothing

        for k, raw_score in enumerate(sorted_scores, 1):
            # Normalise to [0.01, 0.99] — avoid 0 or 1 extremes
            p_item = 0.01 + 0.98 * ((raw_score - min_s) / spread)
            p_none *= (1.0 - p_item)
            recall = 1.0 - p_none
            precision = recall / k if k > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            if f1 > best_f1:
                best_f1, best_k = f1, k
            if k > 50:   # Early stop — no need to go past 50
                break

        return max(best_k, min_k)

    # ── Public API ───────────────────────────────────────────────────────────

    def recommend(
        self,
        cart_product_ids: List[int],
        top_k: int = 10,
        n_retrieve: int = 100,
        use_f1_cutoff: bool = True,
        recency_boost: float = 0.05,
        popularity_penalty: float = 0.3,
    ) -> dict:
        """
        Generate personalised recommendations for the current cart.

        Args:
            cart_product_ids   : Product IDs currently in the user's basket.
            top_k              : Max number of recommendations to return.
            n_retrieve         : Number of similar baskets to retrieve from Faiss.
            use_f1_cutoff      : Whether to apply Faron's F1 optimal K selection.
            recency_boost      : Weight to boost trending/popular products slightly.
            popularity_penalty : Penalty to reduce \"global top-sellers\" dominance.

        Returns:
            dict with cart info, recommendations, and retrieval metadata.
        """
        if not cart_product_ids:
            return {"error": "Cart is empty.", "recommendations": []}

        # 1. Encode cart → query vector
        query_vec = self._encode_cart(cart_product_ids)

        # 2. Retrieve similar baskets (with MMR diversity)
        retrieved = self._retrieve(query_vec, n_retrieve=n_retrieve)

        # 3. Score candidate products
        scored = self._score_candidates(
            retrieved, cart_product_ids,
            recency_boost=recency_boost,
            popularity_penalty=popularity_penalty,
        )

        # 4. Determine optimal K via Faron's F1 if requested
        if use_f1_cutoff:
            optimal_k = self._faron_f1_cutoff(scored)
            final_k   = min(optimal_k, top_k)
        else:
            final_k = top_k

        # 5. Sort and truncate
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:final_k]

        recommendations = [
            {
                "rank":         i + 1,
                "product_id":   pid,
                "product_name": self.id2name.get(pid, f"Product {pid}"),
                "score":        round(score, 6),
            }
            for i, (pid, score) in enumerate(ranked)
        ]

        return {
            "cart_products": [
                {
                    "product_id":   pid,
                    "product_name": self.id2name.get(pid, f"Product {pid}"),
                }
                for pid in cart_product_ids
            ],
            "recommendations": recommendations,
            "model":           "Basket-RAG",
            "retrieval_stats": {
                "baskets_retrieved": len(retrieved),
                "candidates_scored": len(scored),
                "final_k":          final_k,
                "f1_cutoff_used":   use_f1_cutoff,
            },
            "params": {
                "top_k":              top_k,
                "n_retrieve":         n_retrieve,
                "recency_boost":      recency_boost,
                "popularity_penalty": popularity_penalty,
            },
        }
