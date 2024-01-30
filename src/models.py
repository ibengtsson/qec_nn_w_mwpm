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


def main():
    
    model = EdgeNet()
    input = torch.ones((100, 20))
    label = torch.ones((100, 100))
    loss_fun = nn.MSELoss()
    
    out = model(input)
    loss = loss_fun(out, label)
    loss.backward()
    
    print(loss)
    
if __name__ == "__main__":
    main()
    
    
        
        
        

