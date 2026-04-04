import urllib.request
import json
import time

queries = [
    "Chocolate Sandwich Cookies", # Exact match test 
    "Chocolet sand wich cookeis", # Typo test for the same product
    "beef burger meat",           # Semantic search
    "baf borger met",             # Semantic search + typo
    "Apple Juse",                 # Fruit / Drink typo
    "organc turkey brger",        # Typo + normalization testing
    "low fat dairy drink"         # Pure semantic test
]

print("Starting Search Engine Tests (V2)...\n" + "-"*40)

for q in queries:
    req = urllib.request.Request("http://localhost:8000/api/search/", 
                                 data=json.dumps({"query": q, "top_k": 3}).encode(), 
                                 headers={'Content-Type': 'application/json'})
    
    try:
        t0 = time.time()
        with urllib.request.urlopen(req) as response:
            t1 = time.time()
            res = json.loads(response.read().decode())
            print(f"[Query]: '{res['query']}' (took {((t1-t0)*1000):.1f}ms)")
            for idx, r in enumerate(res['results']):
                print(f"  {idx+1}. {r['product_name']} (Score: {r['score']:.4f})")
            print("-" * 40)
    except Exception as e:
        print(f"\nFailed query '{q}': {e}")
