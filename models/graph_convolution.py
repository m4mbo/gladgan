import sys
sys.path.append('../')  

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.graph_processing import normalize_adj

class GraphConv(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 dropout: float=0.0, 
                 bias: bool=True,
                 add_self: bool=True,
                 normalize_embedding: bool=True,
                 expect_normal: bool=False,
                 device: torch.device=None):
               
        super(GraphConv, self).__init__()

        self.expect_normal = expect_normal
        self.add_self = add_self
        self.dropout = dropout
        self.normalize_embedding = normalize_embedding

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)

        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).to(self.device))
        nn.init.xavier_uniform_(self.weight)
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).to(self.device))
            nn.init.zeros_(self.bias)
        else:
            self.bias = None

    def forward(self, x, adj):  

        x.to(self.device)
        adj.to(self.device)
        
        if not self.expect_normal:
            adj = normalize_adj(adj)

        if self.dropout > 0.001:
            x = self.dropout_layer(x)

        y = torch.matmul(adj, x)

        if self.add_self:
            y += x

        y = torch.matmul(y,self.weight)

        if self.bias is not None:
            y = y + self.bias

        if self.normalize_embedding:
            if y.dim() == 3:
                y = F.normalize(y, p=2, dim=2) # if batch dimension is present
            elif y.dim() == 2:
                y = F.normalize(y, p=2, dim=1)

        return y
    
class GraphConv_(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: list,
                 dropout: float=0.0
                ):
        
        """
        Convolution consisting of linear layers
        """
            
        super(GraphConv_, self).__init__()

        self.linear1 = nn.Linear(input_dim, output_dim[0])
        self.linear2 = nn.Linear(output_dim[0], output_dim[1])
        
        self.dropout = dropout

        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, adj, activation=None):
        hidden = torch.stack([self.linear1(x) for _ in range(adj.size(1))], 1)
        hidden = torch.einsum('bijk,bikl->bijl', (adj, hidden))
        hidden = torch.sum(hidden, 1) + self.linear1(input)
        hidden = activation(hidden) if activation is not None else hidden
        
        if self.dropout > 0.001:
            hidden = self.dropout(hidden)

        output = torch.stack([self.linear2(hidden) for _ in range(adj.size(1))], 1)
        output = torch.einsum('bijk,bikl->bijl', (adj, output))
        output = torch.sum(output, 1) + self.linear2(hidden)
        output = activation(output) if activation is not None else output
        output = self.dropout(output)

        return output