import networkx as nx
import numpy as np
import torch.nn.functional as F
import torch

def node_iter(G):
    if float(nx.__version__)<2.0:
        return G.nodes()
    else:
        return G.nodes

def node_dict(G):
    if float(nx.__version__)>2.1:
        node_dict = G.nodes
    else:
        node_dict = G.node
    return node_dict

def process_adj(adj, method, temperature=1.0):
    """
    Process the adjacency matrix logits based on the specified method.
    """
    if method == 'soft':
        # gumbel-softmax without hard threshold
        adj = F.gumbel_softmax(adj.view(-1, 2) / temperature, hard=False).view(adj.size()[:-1])
    elif method == 'hard':
        # gumbel-softmax with hard threshold to binarize
        adj = F.gumbel_softmax(adj.view(-1, 2) / temperature, hard=True).view(adj.size()[:-1])
    else:
        adj = adj
        
    return adj

def normalize_adj(adj, eps=1e-9):
    """
    Row normalize adjacency matrix with epsilon to prevent division by zero.
    """
    if adj.dim() == 2:
        # single adjacency matrix case
        rowsum = adj.sum(1)  # shape: (num_nodes)
        r_inv = (rowsum + eps).pow(-1)  # add epsilon to rowsum to prevent division by zero
        r_mat_inv = torch.diag(r_inv)  
        adj_normalized = torch.matmul(r_mat_inv, adj)  
    elif adj.dim() == 3:
        # batch of adjacency matrices case
        rowsum = adj.sum(2)  # shape: (batch_size, num_nodes)
        r_inv = (rowsum + eps).pow(-1)  # add epsilon to rowsum to prevent division by zero
        r_mat_inv = torch.stack([torch.diag(r) for r in r_inv])  
        adj_normalized = torch.bmm(r_mat_inv, adj) 

    return adj_normalized
