import torch
import torch.nn as nn
import torch_geometric.nn as nng
import numpy as np
import sys
sys.path.append("../")
from src.simulations import SurfaceCodeSim
from src.graph import get_batch_of_graphs
from src.models import MWPMLoss


class SplitSyndromes(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, edges, edge_attr, detector_labels):

        node_range = torch.arange(0, detector_labels.shape[0])
        node_subset = node_range[detector_labels]

        valid_labels = torch.isin(edges, node_subset).sum(dim=0) == 2
        return edges[:, valid_labels], edge_attr[valid_labels, :]


class EdgeConv(nn.Module):

    def __init__(self, n_heads=1, edge_dimensions=2):
        super().__init__()

        self.gat1 = nng.GATConv(
            5,
            16,
            heads=n_heads,
            concat=False,
            edge_dim=edge_dimensions,
            add_self_loops=False,
        )
        self.lin1 = nn.Linear(34, 64)
        self.lin2 = nn.Linear(64, 2)

        self.split_syndromes = SplitSyndromes()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edges, edge_attr, detector_labels):

        x = self.gat1(x, edges, edge_attr)
        x = torch.nn.functional.relu(x, inplace=True)
        x_src, x_dst = x[edges[0, :]], x[edges[1, :]]
        
        edge_feat = torch.cat([x_src, edge_attr, x_dst], dim=-1)
        edge_feat = self.lin1(edge_feat)
        edge_feat = torch.nn.functional.relu(edge_feat, inplace=True)
        edge_feat = self.lin2(edge_feat)
        edges, edge_feat = self.split_syndromes(edges, edge_feat, detector_labels)
        edge_feat = self.sigmoid(edge_feat)
        return edges, edge_feat
def main():
    
    reps = 3
    code_sz = 3
    p = 1e-3
    n_shots = 1000
    sim = SurfaceCodeSim(reps, code_sz, p, n_shots, seed=1)
    n_epochs = 5
    n_batches = 40
    factor = 0.5
    
    model = EdgeConv()
    model.train()
    # loss_fun = nn.MSELoss()
    loss_fun = MWPMLoss.apply
    optim = torch.optim.SGD(model.parameters(), lr=0.001)
    
    w = model.state_dict()["lin2.bias"]
    for epoch in range(n_epochs):
        train_loss = 0
        epoch_n_graphs = 0
        print(list(model.parameters())[0].grad)
        for _ in range(n_batches):
            optim.zero_grad()
            syndromes, flips, n_trivial = sim.generate_syndromes(n_shots)
            x, edges, edge_attr, batch_labels, detector_labels = get_batch_of_graphs(syndromes, 20, code_sz)

            edges, edge_feat = model(x, edges, edge_attr, detector_labels)
            
            node_range = torch.arange(0, x.shape[0])
            loss = loss_fun(
            edges,
            edge_feat,
            batch_labels,
            node_range,
            np.array(flips) * 1,
            factor
            )
    
            loss.backward()
            optim.step()
            n_graphs = syndromes.shape[0]
            train_loss += loss.item() * n_graphs
            epoch_n_graphs += n_graphs
        train_loss /= epoch_n_graphs
    
        print(train_loss)

if __name__ == "__main__":
    main()