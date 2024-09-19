from models.layers import *
import torch
from typing import Callable, Optional
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, 
                 conv_dim: list, 
                 z_dim: int, 
                 num_nodes: int, 
                 num_features: int, 
                 dropout: float=.0):
        
        super(Generator, self).__init__()

        self.num_nodes = num_nodes
        self.num_features = num_features
 
        layers = []
        for c0, c1 in zip([z_dim] + conv_dim[:-1], conv_dim):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.BatchNorm1d(c1))  # do not apply to critic
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout))
        self.multidense_layer = nn.Sequential(*layers)

        self.adj_layer = nn.Linear(conv_dim[-1], num_nodes * num_nodes)
        self.x_layer = nn.Linear(conv_dim[-1], num_nodes * num_features)

        self.dropout = dropout

        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, z):
        
        output = self.multidense_layer(z)

        # shape [batch_size, num_nodes, num_nodes]
        adj_logits = self.adj_layer(output).view(-1, self.num_nodes, self.num_nodes)

        # make the adjacency matrix symmetric by averaging it with its transpose
        adj_logits = (adj_logits + adj_logits.transpose(1, 2)) / 2
        if self.dropout > 0.001:
            adj_logits = self.dropout_layer(adj_logits)
            
        # shape [batch_size, num_nodes, num_features]
        x_logits = self.x_layer(output).view(-1, self.num_nodes, self.num_features)
        if self.dropout > 0.001:
            x_logits = self.dropout_layer(x_logits)

        return adj_logits, x_logits



class Discriminator(nn.Module):
    def __init__(self, 
                 conv_dim: list,
                 aggr_dim: int, 
                 linear_dim: list,
                 device: torch.device,
                 dropout: float=.0,
                 activation: Optional[Callable[[torch.Tensor], torch.Tensor]]=torch.sigmoid):
    
        super(Discriminator, self).__init__()

        gc_input_dim, gc_output_dim = conv_dim    # gc_output_dim is a list

        self.gcn_layer = GraphConv_(gc_input_dim, gc_output_dim, dropout=dropout, device=device)
        self.agg_layer = GraphAggr(gc_output_dim[-1]+gc_input_dim, aggr_dim, dropout=dropout)

        # multi dense layer
        layers = []
        for c0, c1 in zip([aggr_dim]+linear_dim[:-1], linear_dim):
            layers.append(nn.Linear(c0,c1))
            # layers.append(nn.LayerNorm(c1))     # layer norm -> drop-in replacement to batch norm
            layers.append(nn.Dropout(dropout))
        self.multidense_layer = nn.Sequential(*layers)

        self.output_layer = nn.Linear(linear_dim[-1], 1)

        self.activation = activation

    def forward(self, x, adj):
        
        hidden = self.gcn_layer(x, adj)
        hidden = torch.cat((hidden, x), -1)
        hidden = self.agg_layer(hidden, torch.tanh)

        hidden = self.multidense_layer(hidden)

        output = self.output_layer(hidden)
        output = self.activation(output)

        return output, hidden

