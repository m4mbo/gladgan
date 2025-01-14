import pandas as pd
import numpy as np
import networkx as nx
import jsonpickle
from pathlib import Path

def log_graph(adj_mat: np.array, 
              gloss: float, 
              dloss: float, 
              epoch: int,
              out_file: str="bin/graphs.parquet"):
    """
    Log a batch of graphs to a file. Each graph is represented as an adjacency matrix.
    """
    records = []
    batch_size = adj_mat.shape[0]
    
    random_indices = np.random.choice(batch_size, size=3, replace=False)
    for i in random_indices:
        single_adj_mat = adj_mat[i]
    
        graph = nx.from_numpy_array(single_adj_mat)
        
        # serialize the graph
        graph_str = jsonpickle.encode(graph)
        
        # create a record for the current graph
        record = {
            "epoch": epoch,
            "gloss": gloss,
            "dloss": dloss,
            "graph_id": i,
            "graph": graph_str
        }
        records.append(record)

    if Path(out_file).exists():
        df = pd.read_parquet(out_file)
        df = pd.concat([df, pd.DataFrame(records)], ignore_index=True)
    else:
        # new df
        df = pd.DataFrame(records)

    df.to_parquet(out_file, index=False)
