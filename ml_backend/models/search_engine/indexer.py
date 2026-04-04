import pandas as pd
import google.generativeai as genai
import faiss
import numpy as np
import pickle
import os
import time
import re
from collections import defaultdict
from dotenv import load_dotenv

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY not found in .env file.")

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # simple unit extraction / normalization
    text = re.sub(r'\b(ml)\b', 'milliliter', text)
    text = re.sub(r'\b(kg|kgs)\b', 'kilogram', text)
    text = re.sub(r'\b(g|gms)\b', 'gram', text)
    text = re.sub(r'\b(ltr|l)\b', 'liter', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    return text.split()

def generate_ngrams(text, n=3):
    text = text.replace(" ", "")
    if len(text) < n:
        return [text] if text else []
    return [text[i:i+n] for i in range(len(text)-n+1)]

def get_gemini_embeddings(texts, batch_size=100):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            # text-embedding-004 is the recommended model
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=batch,
                task_type="retrieval_document"
            )
            embeddings.extend(result['embedding'])
            print(f"Embedded {len(embeddings)}/{len(texts)}...")
            # Slight sleep to avoid aggressive rate limiting on free tier
            time.sleep(0.5) 
        except Exception as e:
            print(f"Error embedding batch {i}: {e}")
            # Wait longer on error (cooldown)
            time.sleep(5)
            # Retry once
            try:
                 result = genai.embed_content(model="models/text-embedding-004", content=batch)
                 embeddings.extend(result['embedding'])
            except:
                 print(f"FAILED batch {i} completely.")
                 # Fill with zeros so indices match
                 dim = 768 # default for text-embedding-004
                 for _ in batch:
                     embeddings.append([0.0] * dim)
    return embeddings

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_path = os.path.join(base_dir, '..', 'Dataset', 'products_with_images.csv')
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=['product_name'])
    print(f"Loaded {len(df)} products.")
    
    product_names = df['product_name'].tolist()
    product_ids = df['product_id'].tolist()
    
    # Pre-normalize for all steps
    normalized_names = [normalize_text(name) for name in product_names]
    
    # 1. Store mappings
    metadata_map = {}
    for idx, row in df.iterrows():
        metadata_map[str(row['product_id'])] = {
            'product_name': row['product_name'],
            'normalized_name': normalized_names[idx],
            'price': row['price'] if 'price' in row else None,
            'image_link': row['image_link'] if 'image_link' in row else None
        }
        
    with open(os.path.join(data_dir, 'metadata_map.pkl'), 'wb') as f:
        pickle.dump(metadata_map, f)
        
    with open(os.path.join(data_dir, 'id_mapping.pkl'), 'wb') as f:
        pickle.dump(product_ids, f)

    # 2. Build N-Gram Inverted Index (Fuzzy Layer)
    print("Building N-gram Inverted Index...")
    ngram_index = defaultdict(set)
    for pos, norm_name in enumerate(normalized_names):
        tokens = tokenize(norm_name)
        for token in tokens:
            ngrams = generate_ngrams(token, n=3)
            for ng in ngrams:
                ngram_index[ng].add(pos)
                
    ngram_index = {k: list(v) for k, v in ngram_index.items()}
    with open(os.path.join(data_dir, 'ngram_index.pkl'), 'wb') as f:
        pickle.dump(ngram_index, f)

    # 3. Build BM25 Index
    print("Building BM25 Index...")
    from rank_bm25 import BM25Okapi
    tokenized_corpus = [tokenize(doc) for doc in normalized_names]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(os.path.join(data_dir, 'bm25_index.pkl'), 'wb') as f:
        pickle.dump(bm25, f)

    # 4. Build Vector Index (Gemini API)
    if not GEMINI_API_KEY:
        print("Skipping Vector Index build – GEMINI_API_KEY missing.")
        return

    print("Building Vector Index via Gemini API (text-embedding-004)...")
    raw_embeddings = get_gemini_embeddings(normalized_names)
    
    embeddings_np = np.array(raw_embeddings).astype('float32')
    dimension = embeddings_np.shape[1]
    
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings_np)
    index.add(embeddings_np)
    
    faiss.write_index(index, os.path.join(data_dir, 'faiss_index.bin'))
    print("Vector Index Built Successfully.")
    
if __name__ == '__main__':
    main()
