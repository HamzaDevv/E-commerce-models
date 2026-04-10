import os
import pickle
from tqdm import tqdm

def slim_metadata():
    data_dir = "data/basket_rag"
    meta_path = os.path.join(data_dir, "basket_metadata.pkl")
    output_path = os.path.join(data_dir, "basket_metadata_slim.pkl")

    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} not found.")
        return

    print(f"Loading metadata from {meta_path}...")
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    print(f"Slimming {len(metadata)} entries...")
    # Original: {'user_id': 1, 'product_ids': [196, 10258, ...]}
    # Slim: Just the list of product_ids [196, 10258, ...]
    # We can store it as a list of lists. The index in the list will match the FAISS vector index.
    
    slim_data = []
    for entry in tqdm(metadata):
        # We only really need the product_ids for the RAG scoring
        slim_data.append(entry['product_ids'])

    print(f"Saving slimmed metadata to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(slim_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    orig_size = os.path.getsize(meta_path) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Metadata Slimmed Successfully. {orig_size:.2f} MB -> {new_size:.2f} MB")

if __name__ == "__main__":
    slim_metadata()
