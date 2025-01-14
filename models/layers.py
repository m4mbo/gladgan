import torch
import torch.nn as nn
from utils.graph_processing import normalize_adj
import torch.nn.functional as F

class GraphAggr(nn.Module):

    def __init__(self, input_dim, output_dim, dropout=.0):
        super(GraphAggr, self).__init__()
        self.sigmoid_linear = nn.Sequential(nn.Linear(input_dim, output_dim),
                                            nn.Sigmoid())
        self.relu_linear = nn.Sequential(nn.Linear(input_dim, output_dim),
                                         nn.LeakyReLU())
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, activation):

        i = self.sigmoid_linear(input)
        j = self.relu_linear(input)

        output = torch.sum(torch.mul(i, j), 1)

        output = activation(output) if activation is not None else output
        output = self.dropout(output)

        return output

    
class GraphConv(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 dropout: float=0.0, 
                 bias: bool=True,
                 add_self: bool=True,
                 normalize_embedding: bool=True,
                 expect_normal: bool=False,
                 device: torch.device=None
                 ):
               
        super(GraphConv, self).__init__()

        self.expect_normal = expect_normal
        self.add_self = add_self
        self.dropout = dropout
        self.normalize_embedding = normalize_embedding

        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)

        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.bias = nn.Parameter(torch.FloatTensor(output_dim)) if bias else None

        if bias:
            nn.init.zeros_(self.bias)


    def forward(self, x, adj):  
        x = x.to(self.device)
        adj = adj.to(self.device)   

        if not self.expect_normal:
            adj_n = normalize_adj(adj)

        adj = adj_n

        if self.dropout > 0.001:
            x = self.dropout_layer(x)

        hidden = self.linear(x)  # linear transformation
        hidden = torch.matmul(adj, hidden)  # aggregation

        if self.add_self:
            hidden += x  

        if self.bias is not None:
            hidden += self.bias

        if self.normalize_embedding:
            if hidden.dim() == 3:
                hidden = F.normalize(hidden, p=2, dim=2, eps=1e-12)
            elif hidden.dim() == 2:
                hidden = F.normalize(hidden, p=2, dim=2, eps=1e-12)

        return hidden

    
class GraphConv_(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: list,
                 dropout: float=0.0,
                 add_self=False, 
                 normalize_embedding=True, 
                 expect_normal=False, 
                 device=None):
        
        """
        2 graph convolution operations
        """
        super(GraphConv_, self).__init__()

        self.conv1 = GraphConv(input_dim, output_dim[0], dropout, 
                               add_self=add_self, 
                               normalize_embedding=normalize_embedding, 
                               expect_normal=expect_normal, 
                               device=device, bias=None)

        self.conv2 = GraphConv(output_dim[0], output_dim[1], dropout, 
                               add_self=add_self, 
                               normalize_embedding=normalize_embedding, 
                               expect_normal=expect_normal, 
                               device=device, bias=None)
        self.dropout = dropout

    def forward(self, x, adj, activation=None):
        hidden = self.conv1(x, adj)
        if activation is not None:
            hidden = activation(hidden)

        output = self.conv2(hidden, adj)
        if activation is not None:
            output = activation(output)

        return output