# This snippet of code is to show you a simple evaluate for VecDB class, but the full evaluation for project on the Notebook shared with you.
import numpy as np
from vec_db import VecDB
import time
from dataclasses import dataclass
from typing import List
from memory_profiler import memory_usage
import os

DIMENSION = 64

@dataclass
class Result:
    run_time: float
    top_k: int
    db_ids: List[int]
    actual_ids: List[int]

def run_queries(db, np_rows, top_k, num_runs):
    results = []
    max_memory=-1
    for i in range(num_runs):
        query = np.random.random((1,64))
        
        memoryBefore=max(memory_usage())
        tic = time.time()
        db_ids = db.retrieve(query, top_k)
        toc = time.time()
        memoryAfter=max(memory_usage())
        memoryOccupied=memoryAfter-memoryBefore
        run_time = toc - tic
        if memoryOccupied > max_memory:
            max_memory=memoryOccupied
        print(f"for query {i} time: {run_time} , memory used: {memoryOccupied}")
        
        tic = time.time()
        actual_ids = np.argsort(np_rows.dot(query.T).T / (np.linalg.norm(np_rows, axis=1) * np.linalg.norm(query)), axis= 1).squeeze().tolist()[::-1]
        #print(actual_ids)
        toc = time.time()
        np_run_time = toc - tic
        
        results.append(Result(run_time, top_k, db_ids, actual_ids))
    print(f"Max memory: {max_memory}")
    return results

def eval(results: List[Result]):
    # scores are negative. So getting 0 is the best score.
    scores = []
    run_time = []
    for res in results:
        run_time.append(res.run_time)
        # case for retrieving number not equal to top_k, score will be the lowest
        if len(set(res.db_ids)) != res.top_k or len(res.db_ids) != res.top_k:
            scores.append( -1 * len(res.actual_ids) * res.top_k)
            continue
        score = 0
        for id in res.db_ids:
            try:
                ind = res.actual_ids.index(id)
                if ind > res.top_k * 3:
                    score -= ind
            except:
                score -= len(res.actual_ids)
        scores.append(score)

    return sum(scores) / len(scores), sum(run_time) / len(run_time)

""" def memory_usage_run_queries(args):
    global results
    memoryBefore = max(memory_usage())
    memoryAfter = memory_usage(proc=(run_queries, args, {}), interval = 1e-3)
    return results, max(memoryAfter) - memoryBefore """

def create_other_DB_size(input_file, output_file, target_rows, embedding_dim = DIMENSION):
    # Configuration
    dtype = 'float32'

    # 1. Determine the shape of the source file
    # We calculate rows based on file size to be safe, or you can hardcode 20_000_000
    file_size_bytes = os.path.getsize(input_file)
    itemsize = np.dtype(dtype).itemsize
    total_rows = file_size_bytes // (embedding_dim * itemsize)

    print(f"Source detected: {total_rows} rows.")

    # 2. Open source in read mode ('r')
    # This uses almost 0 RAM, it just points to the file on disk
    source_memmap = np.memmap(
        input_file,
        dtype=dtype,
        mode='r',
        shape=(total_rows, embedding_dim)
    )

    # 3. Create the new file in write mode ('w+')
    # We define the shape as the target size (1M, 64)
    dest_memmap = np.memmap(
        output_file,
        dtype=dtype,
        mode='w+',
        shape=(target_rows, embedding_dim)
    )

    # 4. Copy the data
    # This transfers the binary blocks directly
    print("Copying data...")
    dest_memmap[:] = source_memmap[:target_rows]

    # 5. Flush to save changes to disk
    dest_memmap.flush()

    print(f"Success! Saved first {target_rows} rows to {output_file}")


if __name__ == "__main__":
    db_s=10**6
    create_other_DB_size(input_file="OpenSubtitles_en_20M_emb_64.dat", output_file="saved_db.dat", target_rows=db_s)
    db = VecDB(db_size=db_s)

    all_db = db.get_all_rows()

    res = run_queries(db, all_db, 5, 10)
    print(eval(res))
    #print(f"memory used: {memory}")