import numpy as np
from typing import Optional, Tuple, List

# Try to import sklearn KMeans, otherwise we'll use a simple kmeans implementation below
try:
    from sklearn.cluster import KMeans
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False


def _l2_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # returns matrix of squared L2 distances between rows of a and rows of b
    # a: (m, d), b: (n, d) -> (m, n)
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    return aa + bb - 2.0 * (a @ b.T)


def _inner_product_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T


def _kmeans_numpy(X: np.ndarray, n_clusters: int, n_iter: int = 20, seed: Optional[int] = None) -> np.ndarray:
    """
    Simple kmeans (not production quality) used as fallback if sklearn is not available.
    Returns centroids as array (n_clusters, dim)
    """
    rng = np.random.default_rng(seed)
    n_samples, dim = X.shape
    # initialize centroids by picking random distinct samples
    if n_clusters >= n_samples:
        # edge case: more clusters than samples -> pad with small noise
        centroids = X.copy()
        while centroids.shape[0] < n_clusters:
            centroids = np.vstack([centroids, X[rng.integers(0, n_samples)] + 1e-6 * rng.standard_normal(dim)])
        return centroids[:n_clusters]
    idx = rng.choice(n_samples, size=n_clusters, replace=False)
    centroids = X[idx].astype(float)
    for _ in range(n_iter):
        # assign
        dists = _l2_distance_matrix(X, centroids)  # (n_samples, n_clusters)
        labels = np.argmin(dists, axis=1)
        # update
        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            members = X[labels == k]
            if len(members) == 0:
                # reinitialize empty centroid
                new_centroids[k] = X[rng.integers(0, n_samples)]
            else:
                new_centroids[k] = members.mean(axis=0)
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return centroids


class IndexIVFFlat:
    """
    Simple IVF-Flat index:
      - fit(X): runs coarse quantizer (kmeans) => centroids
      - add(x, ids): assign vectors to centroids and store them in lists
      - search(q, top_k, nprobe): find nprobe nearest centroids to q, brute-force search inside those lists, return top_k results
    """
    def __init__(self, dim: int, nlist: int = 64, metric: str = "l2", seed: Optional[int] = None):
        assert metric in ("l2", "ip"), "metric must be 'l2' (Euclidean) or 'ip' (inner product)"
        self.dim = dim
        self.nlist = nlist
        self.metric = metric
        self.seed = seed

        # After fit:
        self.centroids: Optional[np.ndarray] = None  # (nlist, dim)
        # For each list (centroid) we store a list of vectors and list of ids
        self.inverted_lists: List[List[np.ndarray]] = [[] for _ in range(nlist)]
        self.inverted_ids: List[List[int]] = [[] for _ in range(nlist)]
        self.ntotal = 0
        self.is_trained = False

    def fit(self, X: np.ndarray, use_sklearn_kmeans: bool = True, kmeans_iters: int = 20):
        """
        Train coarse quantizer on X (shape: (N, dim))
        """
        assert X.ndim == 2 and X.shape[1] == self.dim
        n_clusters = self.nlist
        if _HAVE_SKLEARN and use_sklearn_kmeans:
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10, max_iter=kmeans_iters)
            kmeans.fit(X)
            centroids = kmeans.cluster_centers_.astype(float)
        else:
            centroids = _kmeans_numpy(X, n_clusters=n_clusters, n_iter=kmeans_iters, seed=self.seed)
        self.centroids = centroids
        # reset inverted lists
        self.inverted_lists = [[] for _ in range(self.nlist)]
        self.inverted_ids = [[] for _ in range(self.nlist)]
        self.ntotal = 0
        self.is_trained = True

    def _assign(self, X: np.ndarray, topk: int = 1) -> np.ndarray:
        """
        Assign each row in X to nearest centroid(s).
        Returns indices of nearest centroids shape (len(X), topk)
        """
        assert self.is_trained and self.centroids is not None
        if self.metric == "l2":
            dmat = _l2_distance_matrix(X, self.centroids)  # squared distances
            idx = np.argsort(dmat, axis=1)[:, :topk]
        else:  # ip
            sim = _inner_product_matrix(X, self.centroids)
            idx = np.argsort(-sim, axis=1)[:, :topk]
        return idx

    def add(self, X: np.ndarray, ids: Optional[np.ndarray] = None):
        """
        Add vectors X (N, dim) with integer ids (N,)
        """
        assert self.is_trained, "Call fit() before add()"
        assert X.ndim == 2 and X.shape[1] == self.dim
        N = X.shape[0]
        if ids is None:
            ids = np.arange(self.ntotal, self.ntotal + N, dtype=int)
        assert len(ids) == N

        assigned = self._assign(X, topk=1).reshape(-1)
        for i in range(N):
            list_no = int(assigned[i])
            # store as numpy arrays in python lists; can be converted to a stacked array for faster search
            self.inverted_lists[list_no].append(X[i].astype(np.float32))
            self.inverted_ids[list_no].append(int(ids[i]))
            self.ntotal += 1

    def _search_in_list(self, q: np.ndarray, list_idx: int, top_k: int) -> List[Tuple[float, int]]:
        """
        Search single list. Returns list of (score, id) of up to top_k matches.
        score: distance (lower better) for l2, similarity (higher better) for ip
        """
        vectors = self.inverted_lists[list_idx]
        ids = self.inverted_ids[list_idx]
        if len(vectors) == 0:
            return []
        X = np.vstack(vectors)  # (m, dim)
        if self.metric == "l2":
            dists = np.sum((X - q) ** 2, axis=1)  # squared distances
            idxs = np.argsort(dists)[:top_k]
            return [(float(dists[i]), int(ids[i])) for i in idxs]
        else:
            sims = X @ q
            idxs = np.argsort(-sims)[:top_k]
            return [(float(sims[i]), int(ids[i])) for i in idxs]

    def search(self, queries: np.ndarray, top_k: int = 10, nprobe: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for each query vector in queries (Q, dim). Returns (distances/sims, ids):
          - distances shape (Q, top_k): for l2 -> sorted ascending (small -> good), ip -> sorted descending (large -> good)
          - ids shape (Q, top_k)
        nprobe: how many coarse clusters to examine (defaults to min(5, nlist))
        """
        assert self.is_trained
        assert queries.ndim == 2 and queries.shape[1] == self.dim
        Q = queries.shape[0]
        if nprobe is None:
            nprobe = min(5, self.nlist)

        # find nearest centroids to each query
        if self.metric == "l2":
            centroid_dists = _l2_distance_matrix(queries, self.centroids)  # (Q, nlist)
            nearest_lists = np.argsort(centroid_dists, axis=1)[:, :nprobe]
        else:
            centroid_sims = _inner_product_matrix(queries, self.centroids)
            nearest_lists = np.argsort(-centroid_sims, axis=1)[:, :nprobe]

        # For each query, search inside selected lists and aggregate
        all_distances = np.full((Q, top_k), np.inf if self.metric == "l2" else -np.inf, dtype=float)
        all_ids = np.full((Q, top_k), -1, dtype=int)

        for qi in range(Q):
            q = queries[qi].astype(np.float32)
            candidates: List[Tuple[float, int]] = []
            for list_idx in nearest_lists[qi]:
                res = self._search_in_list(q, int(list_idx), top_k)
                candidates.extend(res)
            if len(candidates) == 0:
                continue
            # aggregate and pick top_k
            if self.metric == "l2":
                candidates.sort(key=lambda x: x[0])  # ascending distance
            else:
                candidates.sort(key=lambda x: -x[0])  # descending similarity
            chosen = candidates[:top_k]
            for k, (score, id_) in enumerate(chosen):
                all_distances[qi, k] = score
                all_ids[qi, k] = id_

        return all_distances, all_ids

    def stats(self) -> dict:
        sizes = [len(lst) for lst in self.inverted_lists]
        return {
            "ntotal": self.ntotal,
            "nlist": self.nlist,
            "min_list_size": int(min(sizes)) if sizes else 0,
            "max_list_size": int(max(sizes)) if sizes else 0,
            "avg_list_size": float(np.mean(sizes)) if sizes else 0.0
        }


if __name__ == "__main__":
    # Quick example / self-test
    rng = np.random.default_rng(1234)
    dim = 16
    N = 5000
    nlist = 64
    X = rng.normal(size=(N, dim)).astype(np.float32)
    ids = np.arange(N)
    # build index
    idx = IndexIVFFlat(dim=dim, nlist=nlist, metric="l2", seed=42)
    print("Training coarse quantizer...")
    idx.fit(X, use_sklearn_kmeans=_HAVE_SKLEARN, kmeans_iters=20)
    print("Adding vectors...")
    idx.add(X, ids=ids)
    print("Index stats:", idx.stats())

    # create queries (some from the dataset, some random)
    queries = X[:5] + 0.01 * rng.normal(size=(5, dim))
    print("Searching...")
    distances, found_ids = idx.search(queries, top_k=5, nprobe=4)
    print("Distances:\n", distances)
    print("IDs:\n", found_ids)
