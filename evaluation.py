# This snippet of code is to show you a simple evaluate for VecDB class, but the full evaluation for project on the Notebook shared with you.
import numpy as np
from vec_db import VecDB
import time
from dataclasses import dataclass
from typing import List
from memory_profiler import memory_usage

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
        query = np.random.random((1,70))
        
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

if __name__ == "__main__":
    db = VecDB(db_size = 2*(10**7))

    all_db = db.get_all_rows()

    res = run_queries(db, all_db, 5, 10)
    print(eval(res))
    #print(f"memory used: {memory}")