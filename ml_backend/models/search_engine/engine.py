import os
import pickle
import numpy as np
import re
import math
from rapidfuzz import fuzz
from collections import defaultdict


class HybridSearchEngine:
    """
    Two-layer search engine: Fuzzy (trigram + RapidFuzz) + BM25.
    No external API dependencies — runs fully offline.
    """

    def __init__(self, data_dir=None):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.data_dir = os.path.join(base_dir, 'data')
        else:
            self.data_dir = data_dir

        self._load_indices()

    def _load_indices(self):
        print("Loading Search Engine Indices (BM25 + Fuzzy)...")
        bm25_path    = os.path.join(self.data_dir, 'bm25_index.pkl')
        id_map_path  = os.path.join(self.data_dir, 'id_mapping.pkl')
        meta_map_path = os.path.join(self.data_dir, 'metadata_map.pkl')
        ngram_path   = os.path.join(self.data_dir, 'ngram_index.pkl')

        for path in [bm25_path, id_map_path, meta_map_path, ngram_path]:
            if not os.path.exists(path):
                raise Exception(f"Index file not found: {path}")

        with open(bm25_path, 'rb') as f:
            self.bm25 = pickle.load(f)

        with open(id_map_path, 'rb') as f:
            self.id_mapping = pickle.load(f)

        with open(meta_map_path, 'rb') as f:
            self.metadata_map = pickle.load(f)

        with open(ngram_path, 'rb') as f:
            self.ngram_index = pickle.load(f)

        print("Indices Loaded Successfully.")

    # ── Text processing ────────────────────────────────────────────────────────

    def normalize_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\b(ml)\b', 'milliliter', text)
        text = re.sub(r'\b(kg|kgs)\b', 'kilogram', text)
        text = re.sub(r'\b(g|gms)\b', 'gram', text)
        text = re.sub(r'\b(ltr|l)\b', 'liter', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text):
        return text.split()

    def generate_ngrams(self, text, n=3):
        text = text.replace(" ", "")
        if len(text) < n:
            return [text] if text else []
        return [text[i:i+n] for i in range(len(text)-n+1)]

    # ── Fuzzy layer ────────────────────────────────────────────────────────────

    def get_fuzzy_candidates(self, norm_query, top_n=100):
        query_ngrams = []
        for token in self.tokenize(norm_query):
            query_ngrams.extend(self.generate_ngrams(token, n=3))

        candidate_counts = defaultdict(int)
        for ng in set(query_ngrams):
            if ng in self.ngram_index:
                for idx in self.ngram_index[ng]:
                    candidate_counts[idx] += 1

        top_candidates = sorted(candidate_counts.items(), key=lambda x: x[1], reverse=True)[:500]

        fuzzy_scores = []
        for idx, _ in top_candidates:
            product_id = self.id_mapping[idx]
            product_norm_name = self.metadata_map[str(product_id)].get('normalized_name', '')
            score = fuzz.token_sort_ratio(norm_query, product_norm_name) / 100.0
            fuzzy_scores.append((idx, score))

        fuzzy_scores.sort(key=lambda x: x[1], reverse=True)
        return fuzzy_scores[:top_n]

    # ── Main search ────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10):
        norm_query   = self.normalize_text(query)
        query_tokens = set(self.tokenize(norm_query))

        if not norm_query:
            return []

        # -- Layer 1: Fuzzy (trigram + token_sort_ratio) --
        fuzzy_candidates = self.get_fuzzy_candidates(norm_query, top_n=100)
        fuzzy_dict = {idx: score for idx, score in fuzzy_candidates}

        # -- Layer 2: BM25 --
        tokenized_query = self.tokenize(norm_query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0
        bm25_top_idx = np.argsort(bm25_scores)[::-1][:100]
        bm25_dict = {i: float(bm25_scores[i] / max_bm25) for i in bm25_top_idx if bm25_scores[i] > 0}

        # -- Combine: 60% fuzzy + 40% BM25 --
        all_candidate_indices = set(fuzzy_dict.keys()) | set(bm25_dict.keys())

        final_scores = []
        for idx in all_candidate_indices:
            f_score = fuzzy_dict.get(idx, 0.0)
            b_score = bm25_dict.get(idx, 0.0)
            combined_score = (0.6 * f_score) + (0.4 * b_score)

            product_id = self.id_mapping[idx]
            product_norm_name = self.metadata_map[str(product_id)].get('normalized_name', '')
            product_tokens = set(self.tokenize(product_norm_name))

            # Exact token overlap bonus
            if query_tokens & product_tokens:
                combined_score += 0.15

            # Prefix match bonus
            for qt in query_tokens:
                if any(pt.startswith(qt) for pt in product_tokens):
                    combined_score += 0.05
                    break

            final_scores.append((product_id, combined_score))

        final_scores.sort(key=lambda x: x[1], reverse=True)
        final_scores = final_scores[:top_k]

        final_results = []
        for doc_id, score in final_scores:
            meta = self.metadata_map.get(str(doc_id), {})

            price = meta.get("price")
            if isinstance(price, float) and math.isnan(price):
                price = None

            image_link = meta.get("image_link")
            if isinstance(image_link, float) and math.isnan(image_link):
                image_link = None
            elif not isinstance(image_link, str):
                image_link = None

            final_results.append({
                "product_id":   doc_id,
                "product_name": meta.get("product_name"),
                "price":        price,
                "image_link":   image_link,
                "score":        score
            })

        return final_results
