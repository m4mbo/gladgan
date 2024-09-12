import networkx as nx
import numpy as np
import torch
import torch.utils.data
from utils.graph_processing import *

class GraphSampler(torch.utils.data.Dataset):
    def __init__(self, G_list, features='default', normalize=True, max_num_nodes=0):
        self.adj_all = []
        self.len_all = []
        self.feature_all = []
        self.label_all = []
        
        self.max_num_nodes = max_num_nodes

        if features == 'default':
            self.feat_dim = node_dict(G_list[0])[0]['feat'].shape[0]

        for G in G_list:
            adj = np.array(nx.to_numpy_array(G))

            if normalize:
                adj = normalize_adj(adj)

            self.adj_all.append(adj)
            self.len_all.append(G.number_of_nodes())
            self.label_all.append(G.graph['label'])

            if features == 'default':
                f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)

                for i,u in enumerate(G.nodes()):
                    f[i,:] = node_dict(G)[u]['feat']
                self.feature_all.append(f)

            # assigning node feature as its degree
            elif features == 'deg':
                degs = np.sum(np.array(adj), 1)

                if self.max_num_nodes > G.number_of_nodes():
                    degs = np.expand_dims(np.pad(degs, (0, self.max_num_nodes - G.number_of_nodes()), 'constant', constant_values=0), axis=1)
                elif self.max_num_nodes < G.number_of_nodes():
                    deg_index = np.argsort(degs, axis=0)
                    deg_ind = deg_index[0: G.number_of_nodes()-self.max_num_nodes]
                    degs = np.delete(degs, [deg_ind], axis=0)
                    degs = np.expand_dims(degs, axis=1)
                else:
                    degs = np.expand_dims(degs, axis=1)                                        
                self.feature_all.append(degs)

        self.feat_dim = self.feature_all[0].shape[1]

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_nodes = adj.shape[0]

        # padding
        if self.max_num_nodes > num_nodes:
            adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
            adj_padded[:num_nodes, :num_nodes] = adj
        elif self.max_num_nodes < num_nodes:
            degs = np.sum(np.array(adj), 1)
            deg_index = np.argsort(degs, axis=0)
            deg_ind = deg_index[0:num_nodes-self.max_num_nodes]
            adj_padded = np.delete(adj, [deg_ind], axis=0)
            adj_padded = np.delete(adj_padded, [deg_ind], axis=1)
        else:
            adj_padded = adj

        return {'adj':adj_padded,
                'feat':self.feature_all[idx].copy(),
                'label':self.label_all[idx],
                'num_nodes': num_nodes
                }

