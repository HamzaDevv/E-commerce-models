import os
import numpy as np
import faiss
import time

def compress_basket_index():
    data_dir = "data/basket_rag"
    vectors_path = os.path.join(data_dir, "basket_vectors.npy")
    output_path = os.path.join(data_dir, "basket_index.faiss")

    if not os.path.exists(vectors_path):
        print(f"Error: {vectors_path} not found.")
        return

    print(f"Loading large vector file: {vectors_path} ...")
    start_time = time.time()
    # Load vectors (1.47 GB)
    vectors = np.load(vectors_path).astype('float32')
    print(f"Loaded {len(vectors)} vectors in {time.time() - start_time:.2f}s")

    # L2 normalize vectors for cosine similarity
    print("Normalizing vectors...")
    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]
    
    # IVF-PQ Compression Params:
    # n_clusters (nlist): Number of Voronoi cells. 1024 is good for 3M vectors.
    # m: Number of sub-quantizers. 16 means each 128d vector becomes 16 bytes.
    # nbits: Number of bits per sub-vector. 8 bits (256 centroids per sub-quantizer).
    nlist = 1024
    m = 16
    nbits = 8

    print(f"Building IVF-PQ index (nlist={nlist}, m={m}, nbits={nbits})...")
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)

    # Train the index on a sample (or all) of the data
    # Training is required for IVF and PQ to learn cluster centers and codebooks.
    print("Training index (this may take a minute)...")
    index.train(vectors)
    
    print("Adding vectors to index...")
    index.add(vectors)

    print(f"Saving compressed index to {output_path}...")
    faiss.write_index(index, output_path)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Basket Index Compressed Successfully. New size: {file_size:.2f} MB")
    print(f"Original size was ~1470 MB. Compression ratio: {1470 / file_size:.1f}x")

if __name__ == "__main__":
    compress_basket_index()
