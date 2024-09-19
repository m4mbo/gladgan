import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import GraphConv

class Encoder(nn.Module):
    """
    Encoder inspired by f-AnoGAN. Takes a real graph to produce a z vector. 
    Used in the gzg_f approach (discriminator guided).
    """

    def __init__(self, 
                 input_dim: int, 
                 linear_dim: list,
                 z_dim: int, 
                 dropout: float=.0):
        super(Encoder, self).__init__()

        self.gcn_layer = GraphConv(input_dim, linear_dim[0], add_self=False)
        
        layers = []
        for c0, c1 in zip(linear_dim[:-1], linear_dim[1:]):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.LeakyReLU()) 
            layers.append(nn.Dropout(dropout))
        
        self.multidense_layer = nn.Sequential(*layers)

        self.output_layer = nn.Linear(linear_dim[-1], z_dim)

    def forward(self, x, adj):
        hidden = self.gcn_layer(x, adj)
        hidden = F.leaky_relu(hidden)
        hidden = self.multidense_layer(hidden)
        hidden = self.output_layer(hidden)

        # sum pooling
        hidden = hidden.sum(dim=1)
        z_logits = torch.tanh(hidden)

        return z_logits
