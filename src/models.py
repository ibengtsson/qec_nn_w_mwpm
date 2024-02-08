from typing import Any
import torch
import torch.nn as nn
import torch_geometric.nn as nng
from scipy.spatial.distance import cdist
import numpy as np
import pymatching

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
    
class MWPMLoss(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, syndrome, label, decoder: pymatching.Matching):
        
        # possible reshape and move to numpy
        # ...
        
        
        
        ctx.save_for_backward(syndrome, label)
        
        
        

class SimpleTest(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        pass
    

        
        
        

