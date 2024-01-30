import torch
import torch.nn as nn
import torch_geometric.nn as nng
from scipy.spatial.distance import cdist
import numpy as np

class EdgeNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.lin = nn.Linear(20, 20)
        
    def forward(self, x):
        x = self.lin(x)
        x_np = x.detach().numpy()
        
        with torch.no_grad():
            out = cdist(x_np, x_np)
        
        y = torch.tensor(out, requires_grad=True, dtype=torch.float32)
        return y


    
        
        
        

