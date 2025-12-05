import numpy as np
from sklearn.cluster import KMeans
import pickle
import os
import random
import heapq

# Helper functions for memmap operations
def save_memmap(filename, array):
    fp = np.memmap(filename, dtype=array.dtype, mode='w+', shape=array.shape)
    fp[:] = array[:]
    fp.flush()
    del fp
    np.save(filename + '.meta', {'shape': array.shape, 'dtype': str(array.dtype)})

def load_memmap(filename, mode='r'):
    meta = np.load(filename + '.meta.npy', allow_pickle=True).item()
    return np.memmap(filename, dtype=meta['dtype'], mode=mode, shape=meta['shape'])

def pack_codes(codes: np.ndarray) -> np.ndarray:
    N, M = codes.shape
    if M % 2 != 0:
        raise ValueError("M must be even for 4-bit packing.")
    
    packed = (codes[:, 0::2] << 4) | (codes[:, 1::2])
    return packed

def unpack_codes(packed_codes: np.ndarray) -> np.ndarray:
    N, M_packed = packed_codes.shape
    M = M_packed * 2
    
    unpacked = np.zeros((N, M), dtype=np.uint8)
    
    unpacked[:, 0::2] = packed_codes >> 4
    
    unpacked[:, 1::2] = packed_codes & 0x0F
    
    return unpacked


def train(ivfflat, M=8, K=256, SAMPLE_RATIO=0.1):
    centroids = ivfflat._load_centroids()
    
    norm_centroids = np.linalg.norm(centroids, axis=1, keepdims=True)
    normalized_centroids = centroids / norm_centroids
    
    D = ivfflat.vecd 
    assert D % M == 0, "DIMENSION must be divisible by M"
    sub_dim = D // M
    
    sub_matrices = [[] for _ in range(M)]
    
    for cluster_id in range(ivfflat.k):
        print(f"Processing cluster {cluster_id + 1}/{ivfflat.k}...")
        
        vector_ids = ivfflat._load_cluster(cluster_id)
        
        if len(vector_ids) == 0:
            continue
        
        # Randomly sample a % of vectors from this cluster
        sample_size = max(1, int(len(vector_ids) * SAMPLE_RATIO))
        sampled_ids = random.sample(list(vector_ids), sample_size)
        
        centroid = normalized_centroids[cluster_id]
        
        for vec_id in sampled_ids:
            vector = ivfflat._getRow(vec_id)
            
            norm_vec = np.linalg.norm(vector)
            if norm_vec > 0:
                normalized_vector = vector / norm_vec
            else:
                normalized_vector = vector
            
            residual = normalized_vector - centroid
            
            for m in range(M):
                subvector = residual[m * sub_dim: (m + 1) * sub_dim]
                sub_matrices[m].append(subvector)
    
    for m in range(M):
        sub_matrices[m] = np.array(sub_matrices[m], dtype=np.float32)
    
    N = len(sub_matrices[0]) 

    codebook = np.zeros((M, K, sub_dim), dtype=np.float32)

    for i in range(M):
        print(f"Training K-means for subspace {i+1}/{M}...")
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        kmeans.fit(sub_matrices[i])
        codebook[i] = kmeans.cluster_centers_.astype(np.float32)

    print(f"Codebook shape: {codebook.shape}")
    
    save_memmap("index.dat/pq_codebook.dat", codebook)
    print("Codebook saved to disk.")
    initialize_database(ivfflat, K)
    

def initialize_database(ivfflat, PQ_K):
    """
    Encode all vectors using PQ codes, processing one cluster at a time.
    """
    codebook = load_memmap("index.dat/pq_codebook.dat")
    
    centroids = ivfflat._load_centroids()
    
    norm_centroids = np.linalg.norm(centroids, axis=1, keepdims=True)
    normalized_centroids = centroids / norm_centroids

    M = codebook.shape[0]
    sub_dim = codebook.shape[2]
    
    N = ivfflat._get_num_records()
    codes = np.zeros((N, M), dtype=np.uint8)
    
    for cluster_id in range(ivfflat.k):
        print(f"Encoding cluster {cluster_id + 1}/{ivfflat.k}...")
        
        vector_ids = ivfflat._load_cluster(cluster_id)
        
        if len(vector_ids) == 0:
            continue
        
        centroid = normalized_centroids[cluster_id]
        
        for vec_id in vector_ids:
            vector = ivfflat._getRow(vec_id)
            
            norm_vec = np.linalg.norm(vector)
            if norm_vec > 0:
                normalized_vector = vector / norm_vec
            else:
                normalized_vector = vector
            
            residual = normalized_vector - centroid
            
            for m in range(M):
                subvector = residual[m * sub_dim: (m + 1) * sub_dim]
                distances = np.linalg.norm(codebook[m] - subvector, axis=1)
                codes[vec_id, m] = np.argmin(distances)
    
    if PQ_K <= 16:
        print(f"Packing codes (M={M} -> {M//2} bytes)...")
        final_codes = pack_codes(codes)
        save_memmap("index.dat/pq_codes.dat", final_codes)
    else:
        print(f"Saving codes directly (PQ_K={PQ_K} uses full 8 bits)...")
        save_memmap("index.dat/pq_codes.dat", codes)

    return codebook

def compute_distance_table(query_vector, centroid_vector, codebook):
    M = codebook.shape[0]
    sub_dim = codebook.shape[2]
    dist_table = np.zeros((M, codebook.shape[1]), dtype=np.float32)

    query_vector = np.asarray(query_vector).flatten()
    centroid_vector = np.asarray(centroid_vector).flatten()
    
    norm_query = np.linalg.norm(query_vector)
    if norm_query > 0:
        query_vector = query_vector / norm_query
    
    norm_centroid = np.linalg.norm(centroid_vector)
    if norm_centroid > 0:
        centroid_vector = centroid_vector / norm_centroid

    query_residual = query_vector - centroid_vector

    for m in range(M):
        subvector = query_residual[m * sub_dim: (m + 1) * sub_dim]
        for k in range(codebook.shape[1]):
            codeword = codebook[m][k]
            dist_table[m][k] = np.linalg.norm(subvector - codeword)
    
    return dist_table


def retrieve(ivfflat, query_vector, nearest_buckets, all_centroids, index_file_path, Z=200):
    codebook = load_memmap(index_file_path + "/pq_codebook.dat")
    PQ_K = codebook.shape[1] 

    # Don't unpack all causes mem leak
    if PQ_K <= 16:
        packed_codes_mmap = load_memmap(index_file_path + "/pq_codes.dat", mode='r') 
    else:
        codes_mmap = load_memmap(index_file_path + "/pq_codes.dat", mode='r')

    current_results = [] 

    for bucket_id in nearest_buckets:
        centroid_vector = all_centroids[bucket_id]
        dist_table = compute_distance_table(query_vector, centroid_vector, codebook)
        vector_ids = ivfflat._load_cluster(bucket_id)
        
        if len(vector_ids) == 0: continue

        if PQ_K <= 16:
            # Just unpack the specific rows
            batch_packed = packed_codes_mmap[vector_ids] 
            batch_codes = unpack_codes(batch_packed)
        else:
            # Just read the specific rows
            batch_codes = codes_mmap[vector_ids]

        for i, vec_id in enumerate(vector_ids):
            code_row = batch_codes[i] 
            dist = 0.0
            
            for m in range(len(code_row)):  
                dist += dist_table[m][code_row[m]]
            
            # 1. If we haven't found Z items yet, just add it.
            if len(current_results) < Z:
                heapq.heappush(current_results, (-dist, vec_id))
            
            else:
                if -dist > current_results[0][0]:
                    #pops the worst and push the new one in one go
                    heapq.heapreplace(current_results, (-dist, vec_id))
        
    # CLEANUP
    if PQ_K > 16:
        del codes_mmap
    else:
        del packed_codes_mmap

    # Turn negative distances back to positive and sort
    final_results = [(-d, vid) for d, vid in current_results]
    final_results.sort(key=lambda x: x[0])
    
    return final_results

def top_k_results(ivfflat, query_vector, nearest_buckets, index_file_path, k=10, Z=200):
    results_heap = []
    current_results = retrieve(ivfflat, query_vector, nearest_buckets, ivfflat._load_centroids(), index_file_path, Z=Z)
    for dist, vec_id in current_results:
        vector = ivfflat._getRow(vec_id)

        dot_product = np.dot(query_vector, vector)
        norm_query = np.linalg.norm(query_vector)
        norm_vector = np.linalg.norm(vector)
        cosine_similarity = dot_product / (norm_query * norm_vector)
        results_heap.append((cosine_similarity, vec_id))
    
    results_heap.sort(key=lambda x: x[0], reverse=True)
    return results_heap[:k]
