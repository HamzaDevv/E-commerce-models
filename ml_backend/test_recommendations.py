import requests
import pickle
import os

# Load the product lookup
lookup_path = 'data/basket/product_lookup.pkl'
with open(lookup_path, 'rb') as f:
    product_lookup = pickle.load(f)

# Helper to find product IDs by keyword
def find_products(keyword, limit=3):
    results = []
    for pid, name in product_lookup.items():
        if keyword.lower() in name.lower():
            results.append((pid, name))
            if len(results) >= limit:
                break
    return results

# Hand-picked product IDs for testing specific scenarios
# (First, let's find the IDs we need)
CEREAL = find_products("cereal", 1)[0][0]
MILK = find_products("whole milk", 1)[0][0]
BANANA = find_products("banana", 1)[0][0] 

DIAPERS = find_products("diapers", 1)[0][0]
WIPES = find_products("wipes", 1)[0][0]

PASTA = find_products("spaghetti", 1)[0][0]
TOMATO_SAUCE = find_products("pasta sauce", 1)[0][0]
CHEESE = find_products("parmesan cheese", 1)[0][0]

CHIPS = find_products("potato chips", 1)[0][0]
SODA = find_products("cola", 1)[0][0]

# Define Test Carts
test_scenarios = {
    "🥞 Breakfast Cart": [CEREAL, MILK, BANANA],
    "👶 Baby Needs Cart": [DIAPERS, WIPES],
    "🍝 Italian Dinner Cart": [PASTA, TOMATO_SAUCE, CHEESE],
    "🍿 Junk Food Cart": [CHIPS, SODA],
}

API_URL = "http://127.0.0.1:8000/api/recommendation/basket-complete"

print("========================================")
print("🧪 BASKETGPT RECOMMENDATION EVALUATION  ")
print("========================================\n")

for scenario_name, cart_ids in test_scenarios.items():
    print(f"--- {scenario_name} ---")
    cart_names = [product_lookup[pid] for pid in cart_ids]
    print(f"🛒 **Cart**: {', '.join(cart_names)}")
    
    payload = {
        "cart_product_ids": cart_ids,
        "num_suggestions": 5,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        
        print("✨ **Suggestions**:")
        for r in data['suggestions']:
            print(f"  -> {r['product_name']} (confidence: {r['confidence']:.3f})")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API Request failed: {e}")
    print()
