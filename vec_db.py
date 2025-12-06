from typing import Dict, List, Annotated
import numpy as np
import os
from ivfflat import IVF

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.ivfflat = IVF(db_path=self.db_path, index_path=self.index_path, vecd=DIMENSION, k=4000, seed=DB_SEED_NUMBER, cpuCores=14, maxRam=50 * 1024 * 1024)
        self.db_size = db_size
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
    """ def __init__(self, database_file_path="saved_db.dat", index_file_path="index.dat", new_db=True, db_size=None, database_embedded_file_path=None):

        self.db_path = database_file_path
        self.index_path = index_file_path
        # If an embedded DB is provided, read exactly db_size rows from it using memmap
        if database_embedded_file_path is not None:
            if db_size is None:
                raise ValueError("db_size must be provided when using database_embedded_file_path")

            if not os.path.exists(database_embedded_file_path):
                raise FileNotFoundError(f"Embedded DB file does not exist: {database_embedded_file_path}")

            # Read data from the embedded DB
            src = np.memmap(database_embedded_file_path, dtype=np.float32, mode="r", shape=(db_size,DIMENSION))

            # Create destination DB file and write the content into it
            dst = np.memmap(self.db_path, dtype=np.float32, mode="w+", shape=(db_size,DIMENSION))
            dst[:] = src[:]

            # Close memmaps
            del src
            del dst
            if new_db:
                self._build_index()
            return

        # If new database creation is required
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide db_size when creating a new database")

            if os.path.exists(self.db_path):
                os.remove(self.db_path)

            self.generate_database(db_size)

        self.db_size = db_size
 """
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        #TODO: might change to call insert in the index, if you need
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
            if self.db_size is None:
                nProbe=10
            elif self.db_size<10**7:
                nProbe=25
            elif self.db_size<2*10**7:
                nProbe=10
            else:
                nProbe=15
            return self.ivfflat.search(query,top_k=top_k, nprobe=nProbe) 
    
    """         
        scores = []
        num_records = self._get_num_records()
        # here we assume that the row number is the ID of each vector
        for row_num in range(num_records):
            vector = self.get_one_row(row_num)
            score = self._cal_score(query, vector)
            scores.append((score, row_num))
        # here we assume that if two rows have the same score, return the lowest ID
        scores = sorted(scores, reverse=True)[:top_k]
        return [s[1] for s in scores] 
        """
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        self.ivfflat.fit()

