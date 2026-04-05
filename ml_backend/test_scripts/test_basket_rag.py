"""
Test script for the Basket-RAG Inference Engine.
Loads the trained encoder + Faiss index and runs recommendation queries.
"""
import sys
import os
import time

# Ensure we can import from ml_backend root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from models.basket_rag.engine import BasketRAGEngine


def divider(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def run_test(engine, name, cart_ids, top_k=8):
    print(f"\n🧪 Test: {name}")
    cart_names = [engine.id2name.get(pid, f"ID:{pid}") for pid in cart_ids]
    print(f"   Cart: {cart_names}")

    start = time.perf_counter()
    result = engine.recommend(cart_product_ids=cart_ids, top_k=top_k)
    elapsed = (time.perf_counter() - start) * 1000

    stats = result['retrieval_stats']
    print(f"   ⏱️  Latency: {elapsed:.1f}ms")
    print(f"   📊 Retrieved {stats['baskets_retrieved']} baskets → "
          f"scored {stats['candidates_scored']} candidates → "
          f"returning {stats['final_k']} items (F1={stats['f1_cutoff_used']})")
    print(f"   Recommendations:")

    for rec in result["recommendations"]:
        print(f"     #{rec['rank']:2d}  {rec['product_name']:<50s}  (score: {rec['score']:.4f})")

    return result


if __name__ == "__main__":
    divider("Booting BasketRAG Engine")
    t0 = time.perf_counter()
    engine = BasketRAGEngine()
    boot_time = time.perf_counter() - t0
    print(f"\n🚀 Engine booted in {boot_time:.2f}s")
    print(f"   Vocab: {engine.vocab_size:,} tokens")
    print(f"   Faiss index: {engine.faiss_index.ntotal:,} vectors")

    # ── Test 1: Toddler Breakfast ─────────────────────────────
    divider("Test 1: Toddler Breakfast")
    # Wheat Chex Cereal, Blueberry Yogurt, Teething Wafers
    run_test(engine, "Toddler Breakfast",
             cart_ids=[28, 9, 63])

    # ── Test 2: Italian Dinner ────────────────────────────────
    divider("Test 2: Italian Dinner")
    # Organic Spaghetti, Roasted Garlic Pasta Sauce, Garlic Parmesan
    run_test(engine, "Italian Dinner",
             cart_ids=[33, 244, 60])

    # ── Test 3: Healthy Smoothie ──────────────────────────────
    divider("Test 3: Healthy Smoothie Ingredients")
    # Banana, Strawberry Yogurt, Vanilla Almond Milk, Spinach
    run_test(engine, "Smoothie Ingredients",
             cart_ids=[426, 132, 432, 76])

    # ── Test 4: Single Item Cold-Start ────────────────────────
    divider("Test 4: Cold-start — Single Item")
    # Just Whole Wheat Bread
    run_test(engine, "Single Item (Bread)",
             cart_ids=[83])

    # ── Test 5: Large Mixed Cart ──────────────────────────────
    divider("Test 5: Large Mixed Grocery Cart")
    # Bread, Eggs (Blueberry smoothie), Almond Milk, Avocado, Banana, Spinach
    run_test(engine, "Mixed Grocery Cart",
             cart_ids=[83, 94, 432, 1374, 426, 76],
             top_k=12)

    divider("All Tests Complete ✅")
