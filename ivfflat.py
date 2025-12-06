import numpy as np
from typing import Optional, Tuple, List
from sklearn.cluster import MiniBatchKMeans
import time
import os
import heapq
import pickle
import pq

M=7
K=256
Z=200
SAMPLE_RATIO=0.1


class IVF:
    def __init__(self, db_path,index_path, vecd, k, seed, cpuCores):
        self.db_path=db_path
        self.index_path=index_path
        self.vecd=vecd
        self.k=k
        self.seed=seed
        self.cores=cpuCores
        self.centroids=[]
        self.clusters=[[] for _ in range(k)]

    def _save_centroids(self, centroids: np.ndarray) -> None:

        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)
        centroids_file_path = os.path.join(self.index_path, "centroids.pkl")
        with open(centroids_file_path, "wb") as centroids_file:
            pickle.dump(centroids, centroids_file)

    def _save_clusters(self, clusters: List[np.ndarray]) -> None:
        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)
        for cluster_id, cluster in enumerate(clusters):
            cluster_file_path = os.path.join(self.index_path, f"cluster_{cluster_id}.npy")
            np.save(cluster_file_path, np.array(cluster, dtype=np.int32))

    def _load_centroids(self) -> np.ndarray:

        centroids_file_path = os.path.join(self.index_path, "centroids.pkl")
        with open(centroids_file_path, "rb") as centroids_file:
            centroids = pickle.load(centroids_file)
        return centroids
    
    def _load_cluster(self, cluster_id: int) -> np.ndarray:
        cluster_file_path = os.path.join(self.index_path, f"cluster_{cluster_id}.npy")
        try:
            cluster = np.load(cluster_file_path)
        except FileNotFoundError:
            return np.array([], dtype=np.int32)
        return cluster

    def _load_clusters(self) -> List[np.ndarray]:
  
        clusters = []
        for cluster_id in range(self.cluster_count):
            cluster = self._load_cluster(cluster_id)
            clusters.append(cluster)
        return clusters
    
    
    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (70 * 4)
    
    def _getRow(self,id):
        offset= id*self.vecd*4
        vector=np.memmap(filename=self.db_path,dtype=np.float32, mode='r', offset=offset, shape=(1,self.vecd))
        return(np.array(vector[0]))
    
    def _getAllRows(self):
        num_records=self._get_num_records()
        vector=np.memmap(filename=self.db_path,dtype=np.float32, mode='r', shape=(num_records,self.vecd))
        return(np.array(vector))

    def fit(self, M=7, PQ_K=256, SAMPLE_RATIO=0.1):
        db=self._getAllRows()
        start=time.time()
        model = MiniBatchKMeans(n_clusters=self.k, random_state=42, batch_size=256*self.cores, max_iter=300)
        model.fit(db)
        end=time.time()
        print(f"Time in learning is {end-start}")

        centroids = model.cluster_centers_.astype(float)
        self._save_centroids(centroids)
        normCentroids=np.linalg.norm(centroids, axis=1, keepdims=True)
        normalizedCentroids=centroids/normCentroids
        #batchSize = floor( (limit_bytes * safety) / (bytes_per_float * (D + n_clusters)) )
        #batchSize= floor ((50MB*0.8)/(4*(70+100)))=60K if n_clusters 64 then 76K
        start=time.time()
        batchSize= 60000
        clusters=[[]for _ in range(self.k)]
        for i in range(0,len(db),batchSize):
            batchedVectors=db[i:i+batchSize]
            normBatch=np.linalg.norm(batchedVectors, axis=1, keepdims=True)
            normalizedBatch=batchedVectors/normBatch
            cosineSimilarties=np.dot(normalizedBatch,normalizedCentroids.T)
            batchLabel=np.argmax(cosineSimilarties,axis=1)
            for id,label in enumerate(batchLabel):
                clusters[label].append(i+id)
        end=time.time()
        print(f"Time in clustering is {end-start}")
        self._save_clusters(clusters)
        pq.train(self, M=M, K=PQ_K, SAMPLE_RATIO=SAMPLE_RATIO)
    
    def compute_similarity_matrix(self,query, centroids):
        if query.ndim == 1:
            query = query[np.newaxis, :]
        querynorm=np.linalg.norm(query, axis=1, keepdims=True)
        normalizedquery=query/querynorm
        normCentroids=np.linalg.norm(centroids, axis=1, keepdims=True)
        normalizedCentroids=centroids/normCentroids
        similarities=np.dot(normalizedquery,normalizedCentroids.T)
        return similarities
    
    def compute_similarity_vector(self,query, vector):
        querynorm=np.linalg.norm(query)
        normalvector=np.linalg.norm(vector)
        if querynorm == 0 or normalvector == 0:
            return 0
        normalizedquery=query/querynorm
        normalizedVector=vector/normalvector
        similarities=np.dot(normalizedquery,normalizedVector)
        return similarities

    def searchInCandidates(self, query, top_k, clustersindexes):
        pairs = []
        for vecid in clustersindexes:
            vector=self._getRow(vecid)
            similarity = self.compute_similarity_vector(query,vector)
            pairs.append((similarity, vecid))
        return heapq.nlargest(top_k,pairs)
    
    def search(self, query, top_k, nprobe, index_file_path, Z=200):
        # Load centroids once and pass them to avoid reloading
        centroids=self._load_centroids()
        similarities = self.compute_similarity_matrix(query, centroids)
        
        topN_indices = np.argpartition(similarities, -nprobe,axis=1)[:, -nprobe:]
        del similarities
        topN_indices_1d = topN_indices.flatten()
        
        results = pq.top_k_results(self, query, topN_indices_1d, centroids, index_file_path, k=top_k, Z=Z)
        
        del centroids
        return [result[1] for result in results]

        # candidateVectorsIds=set()
        # for i in topN_indices_1d:
        #         cluster=self._load_cluster(i)
        #         for j in cluster:
        #             candidateVectorsIds.add(j)
        # pairs=self.searchInCandidates(query=query, top_k=top_k, clustersindexes=candidateVectorsIds)
        # return [result[1] for result in pairs]
        

  

# if __name__ == "__main__":
#     rng = np.random.default_rng(1234)
#     vecd = 70
#     N = 1000000
#     nlist = 100
#     X = rng.normal(size=(N, vecd)).astype(np.float32)
#     ids = np.arange(N)
#     # build index
#     ivf = IVF(db_path="",vecd=vecd, k=nlist, seed=42,cpuCores=14)
#     print("Training coarse quantizer...")
#     ivf.fit()

#     # create queries (some from the dataset, some random)
#     queries = X[0] + 0.01 * rng.normal(size=(1, vecd))
#     print("Searching...")
#     distances, found_ids = ivf.search(queries, top_k=5, nprobe=4)
#     print("Distances:\n", distances)
#     print("IDs:\n", found_ids) 