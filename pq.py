import numpy as np
from sklearn.cluster import KMeans
import pickle
import os
import random

N_CLUSTERS = 1024    
DIMENSION = 64       
N_SAMPLES = 100000

# Helper functions for memmap operations
def save_memmap(filename, array):
    """Save array to a raw binary file using memmap for best performance."""
    fp = np.memmap(filename, dtype=array.dtype, mode='w+', shape=array.shape)
    fp[:] = array[:]
    fp.flush()
    del fp
    # Save metadata (shape and dtype) separately
    np.save(filename + '.meta', {'shape': array.shape, 'dtype': str(array.dtype)})

def load_memmap(filename, mode='r'):
    """Load array from a raw binary file using memmap for best performance."""
    meta = np.load(filename + '.meta.npy', allow_pickle=True).item()
    return np.memmap(filename, dtype=meta['dtype'], mode=mode, shape=meta['shape'])

# Loads one cluster at a time to avoid memory issues
# Calculate the Residuals
# Codebook Size: (M, K, sub_dim)
def train(ivfflat, M=8, K=256):
    """
    Train PQ codebook by processing one cluster at a time.
    ivfflat: an IVF object that has _load_centroids, _load_cluster, k, vecd, and _getRow methods
    """
    # Load centroids using IVF object's method
    centroids = ivfflat._load_centroids()
    
    # Normalize centroids once
    norm_centroids = np.linalg.norm(centroids, axis=1, keepdims=True)
    normalized_centroids = centroids / norm_centroids
    
    D = ivfflat.vecd  # vector dimension
    assert D % M == 0, "DIMENSION must be divisible by M"
    sub_dim = D // M
    
    # Collect sub-vectors for each subspace across all clusters
    # sub_matrices[m] will hold all sub-vectors for subspace m
    sub_matrices = [[] for _ in range(M)]
    
    # Process one cluster at a time
    for cluster_id in range(ivfflat.k):
        print(f"Processing cluster {cluster_id + 1}/{ivfflat.k}...")
        
        # Load vector IDs in this cluster using IVF object's method
        vector_ids = ivfflat._load_cluster(cluster_id)
        
        if len(vector_ids) == 0:
            continue
        
        # Randomly sample 10% of vectors from this cluster
        sample_size = max(1, int(len(vector_ids) * 0.1))
        sampled_ids = random.sample(list(vector_ids), sample_size)
        
        # Get the centroid for this cluster (already normalized)
        centroid = normalized_centroids[cluster_id]
        
        # Process each sampled vector in the cluster
        for vec_id in sampled_ids:
            # Load vector from disk using IVF object's method
            vector = ivfflat._getRow(vec_id)
            
            # Normalize vector
            norm_vec = np.linalg.norm(vector)
            if norm_vec > 0:
                normalized_vector = vector / norm_vec
            else:
                normalized_vector = vector
            
            # Compute residual
            residual = normalized_vector - centroid
            
            # Split residual into M sub-vectors and append to sub_matrices
            for m in range(M):
                subvector = residual[m * sub_dim: (m + 1) * sub_dim]
                sub_matrices[m].append(subvector)
    
    # Convert lists to numpy arrays
    for m in range(M):
        sub_matrices[m] = np.array(sub_matrices[m], dtype=np.float32)
    
    print(f"Total vectors processed: {len(sub_matrices[0])}")
    N = len(sub_matrices[0]) 

    # Apply K-means clustering to each sub-matrix
    # Codebook shape will be (M, K, sub_dim)
    codebook = np.zeros((M, K, sub_dim), dtype=np.float32)

    for i in range(M):
        print(f"Training K-means for subspace {i+1}/{M}...")
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        kmeans.fit(sub_matrices[i])
        codebook[i] = kmeans.cluster_centers_.astype(np.float32)

    print(f"Codebook shape: {codebook.shape}")
    
    # Save codebook to disk using memmap for best performance
    save_memmap("pq_codebook.dat", codebook)
    print("Codebook saved to disk.")
    initialize_database(ivfflat)
    

def initialize_database(ivfflat):
    """
    Encode all vectors using PQ codes, processing one cluster at a time.
    ivfflat: an IVF object that has _load_centroids, _load_cluster, k, vecd, and _getRow methods
    """
    # Load codebook from disk (using memmap for best performance)
    codebook = load_memmap("pq_codebook.dat")
    
    # Load centroids
    centroids = ivfflat._load_centroids()
    
    # Normalize centroids once
    norm_centroids = np.linalg.norm(centroids, axis=1, keepdims=True)
    normalized_centroids = centroids / norm_centroids

    M = codebook.shape[0]
    sub_dim = codebook.shape[2]
    
    # Get total number of vectors to initialize codes array
    N = ivfflat._get_num_records()
    codes = np.zeros((N, M), dtype=np.uint8)
    
    # Process one cluster at a time
    for cluster_id in range(ivfflat.k):
        print(f"Encoding cluster {cluster_id + 1}/{ivfflat.k}...")
        
        # Load vector IDs in this cluster
        vector_ids = ivfflat._load_cluster(cluster_id)
        
        if len(vector_ids) == 0:
            continue
        
        # Get the centroid for this cluster (already normalized)
        centroid = normalized_centroids[cluster_id]
        
        # Process each vector in the cluster
        for vec_id in vector_ids:
            # Load vector from disk
            vector = ivfflat._getRow(vec_id)
            
            # Normalize vector
            norm_vec = np.linalg.norm(vector)
            if norm_vec > 0:
                normalized_vector = vector / norm_vec
            else:
                normalized_vector = vector
            
            # Compute residual
            residual = normalized_vector - centroid
            
            # Encode residual using PQ codebook
            for m in range(M):
                subvector = residual[m * sub_dim: (m + 1) * sub_dim]
                # Find nearest codeword in codebook[m]
                distances = np.linalg.norm(codebook[m] - subvector, axis=1)
                codes[vec_id, m] = np.argmin(distances)
    
    # Save codes once after processing all vectors
    save_memmap("pq_codes.dat", codes)
    print(f"PQ codes saved. Shape: {codes.shape}")

    return codebook

def compute_distance_table(query_vector, centroid_vector, codebook):
    M = codebook.shape[0]
    sub_dim = codebook.shape[2]
    dist_table = np.zeros((M, codebook.shape[1]), dtype=np.float32)

    # Ensure query_vector and centroid_vector are 1D
    query_vector = np.asarray(query_vector).flatten()
    centroid_vector = np.asarray(centroid_vector).flatten()
    
    # Normalize query and centroid
    norm_query = np.linalg.norm(query_vector)
    if norm_query > 0:
        query_vector = query_vector / norm_query
    
    norm_centroid = np.linalg.norm(centroid_vector)
    if norm_centroid > 0:
        centroid_vector = centroid_vector / norm_centroid

    # Compute the residual for the query
    query_residual = query_vector - centroid_vector

    for m in range(M):
        subvector = query_residual[m * sub_dim: (m + 1) * sub_dim]
        for k in range(codebook.shape[1]):
            codeword = codebook[m][k]
            dist_table[m][k] = np.linalg.norm(subvector - codeword)
    
    return dist_table

def load_bucket_from_disk(bucket_id):
    # Placeholder function to simulate loading codes from disk for a specific bucket
    # In practice, this would read from a file or database
    # Here we return a list of (doc_id, pq_code) tuples
    stored_codes = []
    num_codes_in_bucket = 100  # Example number of codes in this bucket
    for i in range(num_codes_in_bucket):
        doc_id = bucket_id * 1000 + i  # Example doc_id
        pq_code = np.random.randint(0, 256, size=(8,), dtype=np.uint8)  # Example PQ code
        stored_codes.append((doc_id, pq_code))
    return stored_codes

def retrieve(ivfflat, query_vector, nearest_buckets, all_centroids, z=200):
    # Load codebook and codes from disk (using memmap for best performance)
    codebook = load_memmap("pq_codebook.dat")
    codes = load_memmap("pq_codes.dat")
    current_results = []  # Placeholder for results heap
    print("Codebook and codes loaded from disk.")
    for bucket_id in nearest_buckets:
        
        # A. Get the centroid for this specific bucket
        centroid_vector = all_centroids[bucket_id]
        
        # B. Calculate the specific Lookup Table for THIS bucket
        #    (Because the residual depends on which centroid we are looking at)
        #    Query Residual = Query - Centroid_Vector
        dist_table = compute_distance_table(query_vector, centroid_vector, codebook)
        
        # C. Load the codes from disk for THIS bucket
        #    (Team 2's job: gives you a list of [ID, Code])
        vector_ids = ivfflat._load_cluster(bucket_id)
        
        # D. The Inner Loop (The "Scan")
        for vec_id in vector_ids:
            
            # Calculate distance using the table
            dist = 0
            for m in range(len(codes[0])):  
                dist += dist_table[m][codes[vec_id][m]]
            
            # E. Add to results heap
            current_results.append((dist, vec_id))
        
        # Momken ne7awel n optimize hena ba3d keda
        current_results.sort(key=lambda x: x[0])
        current_results = current_results[:z]
    return current_results

def top_k_results(ivfflat, query_vector, nearest_buckets, k=10):
    results_heap = []
    current_results = retrieve(ivfflat, query_vector, nearest_buckets, ivfflat._load_centroids())
    # we will use get one row beta3et vec_db
    for dist, vec_id in current_results:
        vector = ivfflat._getRow(vec_id)
        # calculate the exact cosine similarity
        dot_product = np.dot(query_vector, vector)
        norm_query = np.linalg.norm(query_vector)
        norm_vector = np.linalg.norm(vector)
        cosine_similarity = dot_product / (norm_query * norm_vector)
        results_heap.append((cosine_similarity, vec_id))
    
    results_heap.sort(key=lambda x: x[0], reverse=True)
    return results_heap[:k]

# def main():
#     # generate_mock_data()
#     train()

# if __name__ == "__main__":
#     main()