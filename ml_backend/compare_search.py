import sys
import os

# Add current dir to path to import models
sys.path.append('.')

from models.search_engine.engine import HybridSearchEngine

engine = HybridSearchEngine(data_dir='data')

queries = [
    "Chocolate Sandwich Cookies", # Exact match 
    "Chocolet sand wich cookeis", # Typo 
    "beef burger meat",           # Semantic mapping
    "low fat dairy drink",        # Pure semantic
]

print("SEARCH ENGINE COMPARISON: WITH VS WITHOUT SEMANTIC (VECTOR)\n" + "="*60)

for q in queries:
    print(f"\nQUERY: '{q}'")
    
    # 1. With Semantic
    res_v = engine.search(q, top_k=2, use_vector=True)
    # 2. Without Semantic
    res_nv = engine.search(q, top_k=2, use_vector=False)
    
    print(f"  [WITH SEMANTIC]")
    for i, r in enumerate(res_v):
        print(f"    {i+1}. {r['product_name']} (Score: {r['score']:.3f})")
        
    print(f"  [WITHOUT SEMANTIC]")
    for i, r in enumerate(res_nv):
        print(f"    {i+1}. {r['product_name']} (Score: {r['score']:.3f})")
    print("-" * 60)
