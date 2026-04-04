import os
import pickle
import google.generativeai as genai
import faiss
import numpy as np
import re
import math
from rapidfuzz import fuzz
from collections import defaultdict
from dotenv import load_dotenv

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class HybridSearchEngine:
    def __init__(self, data_dir=None):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.data_dir = os.path.join(base_dir, 'data')
        else:
            self.data_dir = data_dir
            
        self._load_indices()

    def _load_indices(self):
        print("Loading Hybrid Search Engine Indices (Gemini V2)...")
        faiss_path = os.path.join(self.data_dir, 'faiss_index.bin')
        bm25_path = os.path.join(self.data_dir, 'bm25_index.pkl')
        id_map_path = os.path.join(self.data_dir, 'id_mapping.pkl')
        meta_map_path = os.path.join(self.data_dir, 'metadata_map.pkl')
        ngram_path = os.path.join(self.data_dir, 'ngram_index.pkl')

        if not os.path.exists(faiss_path) or not os.path.exists(ngram_path):
            raise Exception("Indices not found. Please run indexer.py first.")
            
        self.vector_index = faiss.read_index(faiss_path)
        
        with open(bm25_path, 'rb') as f:
            self.bm25 = pickle.load(f)
            
        with open(id_map_path, 'rb') as f:
            self.id_mapping = pickle.load(f)
            
        with open(meta_map_path, 'rb') as f:
            self.metadata_map = pickle.load(f)
            
        with open(ngram_path, 'rb') as f:
            self.ngram_index = pickle.load(f)
            
        print("Indices Loaded Successfully.")

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

    def search(self, query: str, top_k: int = 10, use_vector: bool = True):
        norm_query = self.normalize_text(query)
        query_tokens = set(self.tokenize(norm_query))
        
        if not norm_query:
            return []

        # -- Layer 1: Fuzzy --
        fuzzy_candidates = self.get_fuzzy_candidates(norm_query, top_n=100)
        fuzzy_dict = {idx: score for idx, score in fuzzy_candidates}

        # -- Layer 2: BM25 --
        tokenized_query = self.tokenize(norm_query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0
        bm25_top_idx = np.argsort(bm25_scores)[::-1][:100]
        bm25_dict = {i: float(bm25_scores[i] / max_bm25) for i in bm25_top_idx if bm25_scores[i] > 0}
        
        # -- Layer 3: Vector (Gemini API) --
        vector_dict = {}
        if use_vector and GEMINI_API_KEY:
            try:
                # Use model 'text-embedding-004'
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=norm_query,
                    task_type="retrieval_query"
                )
                query_emb = np.array(result['embedding']).astype('float32').reshape(1, -1)
                faiss.normalize_L2(query_emb)
                distances, indices = self.vector_index.search(query_emb, 100)
                vector_dict = {indices[0][i]: float(distances[0][i]) for i in range(len(indices[0]))}
            except Exception as e:
                print(f"Gemini Vector search failed: {e}")
                use_vector = False

        all_candidate_indices = set(fuzzy_dict.keys()).union(set(bm25_dict.keys())).union(set(vector_dict.keys()))
        
        final_scores = []
        for idx in all_candidate_indices:
            f_score = fuzzy_dict.get(idx, 0.0)
            b_score = bm25_dict.get(idx, 0.0)
            v_score = vector_dict.get(idx, 0.0)
            
            if use_vector:
                combined_score = (0.4 * f_score) + (0.35 * v_score) + (0.25 * b_score)
            else:
                combined_score = (0.6 * f_score) + (0.4 * b_score)
            
            product_id = self.id_mapping[idx]
            product_norm_name = self.metadata_map[str(product_id)].get('normalized_name', '')
            product_tokens = set(self.tokenize(product_norm_name))
            
            if query_tokens & product_tokens:
                 combined_score += 0.15
                 
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
                "product_id": doc_id,
                "product_name": meta.get("product_name"),
                "price": price,
                "image_link": image_link,
                "score": score
            })
            
        return final_results
