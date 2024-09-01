import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from graph_convolution import GraphConv, GraphConv_
from graph_aggregation import GraphAggr

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, conv_dim, z_dim, num_nodes, num_features, dropout):
        super(Generator, self).__init__()

        self.num_nodes = num_nodes
        self.num_features = num_features
 
        layers = []
        for c0, c1 in zip([z_dim] + conv_dim[:-1], conv_dim):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout, inplace=True))
        self.multidense_layer = nn.Sequential(*layers)

        self.edges_layer = nn.Linear(conv_dim[-1], num_nodes * num_nodes)
        self.nodes_layer = nn.Linear(conv_dim[-1], num_nodes * num_features)

        self.dropout = dropout

        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, z):

        output = self.multidense_layer(z)

        # shape [batch_size, num_nodes, num_nodes]
        edges_logits = self.edges_layer(output).view(-1, self.num_nodes, self.num_nodes)

        # make the adjacency matrix symmetric by averaging it with its transpose
        edges_logits = (edges_logits + edges_logits.transpose(1, 2)) / 2
        if self.dropout > 0.001:
            edges_logits = self.dropout_layer(edges_logits)
            
        # shape [batch_size, num_nodes, num_features]
        nodes_logits = self.nodes_layer(output).view(-1, self.num_nodes, self.num_features)
        if self.dropout > 0.001:
            nodes_logits = self.dropout_layer(nodes_logits)

        return edges_logits, nodes_logits



class Discriminator(nn.Module):
    def __init__(self, 
                 gc_dim: list,
                 aggr_dim: int, 
                 linear_dim: list,
                 dropout: float=0.0):
    
        super(Discriminator, self).__init__()

        gc_input_dim, gc_output_dim = gc_dim    # gc_output_dim is a list

        self.gcn_layer = GraphConv_(gc_input_dim, gc_output_dim, dropout=dropout)
        self.agg_layer = GraphAggr(gc_output_dim[-1], aggr_dim, dropout)

        # multi dense layer
        layers = []
        for c0, c1 in zip([aggr_dim]+linear_dim[:-1], linear_dim):
            layers.append(nn.Linear(c0,c1))
            layers.append(nn.Dropout(dropout))
        self.multidense_layer = nn.Sequential(*layers)

        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, x, adj, activatation=None):
        
        hidden = self.gcn_layer(x, adj)
        hidden = torch.cat((hidden, x), -1)
        hidden = self.agg_layer(hidden, torch.tanh)
        hidden = self.multidense_layer(hidden)

        output = self.output_layer(hidden)
        output = activatation(output) if activatation is not None else output

        return output, hidden

