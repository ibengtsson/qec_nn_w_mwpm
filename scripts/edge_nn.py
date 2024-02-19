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
    
class GraphNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.gc = nng.GraphConv(5, 16)
        self.split_syndromes = SplitSyndromes()
        
        self.lin = nn.Linear(33, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, edges, edge_attr, detector_labels):

        x = self.gc(x, edges, edge_attr[:, 1])
        x = torch.nn.functional.tanh(x)
        
        edges, edge_attr = self.split_syndromes(edges, edge_attr, detector_labels)
        x_src, x_dst = x[edges[0, :]], x[edges[1, :]]
        edge_feat = torch.cat([x_src, edge_attr[:, 0][:, None], x_dst], dim=-1)
        edge_feat = self.lin(edge_feat)
        
        # save edges with largest weights
        n_edges = edge_feat.shape[0]
        edge_feat = edge_feat.reshape(-1,  n_edges // 2)
        edge_classes = edge_attr[:, 1].reshape(-1, n_edges // 2)
        max_inds = torch.argmin(edge_feat, dim=0)

        edge_feat = edge_feat[max_inds, range(n_edges // 2)]
        edge_classes = edge_classes[max_inds, range(n_edges // 2)]
        
        edges = edges[:, :n_edges // 2]

        # edge_feat = self.sigmoid(edge_feat)
        return edges, edge_feat, edge_classes
    
def main():
    
    reps = 5
    code_sz = 5
    p = 1e-3
    n_shots = 1000
    sim = SurfaceCodeSim(reps, code_sz, p, n_shots)
    n_epochs = 10
    n_batches = 10
    factor = 1.5
    # factor = 3
    
    # set seed (for nn, not simulations)
    torch.manual_seed(111)
    model = GraphNN()
    model.train()
    loss_fun = MWPMLoss.apply
    
    optim = torch.optim.SGD(model.parameters(), lr=0.001)
    
    train_losses = []
    for epoch in range(n_epochs):
        train_loss = 0
        epoch_n_graphs = 0
        
        if epoch > 0:
            params = list(model.parameters())
            for i, p in enumerate(params):
                print(f"Parameter {i}: {torch.count_nonzero(p.grad).item()} non-zero gradients.")

        for _ in range(n_batches):
            optim.zero_grad()
            syndromes, flips, n_trivial = sim.generate_syndromes(n_shots)
            x, edges, edge_attr, batch_labels, detector_labels = get_batch_of_graphs(syndromes, m_nearest_nodes=20)
            edges, edge_weights, edge_classes = model(x, edges, edge_attr, detector_labels)
            loss = loss_fun(
                edges,
                edge_weights,
                edge_classes,
                batch_labels,
                flips,
                factor
                )
    
            loss.backward()
            optim.step()
            n_graphs = syndromes.shape[0]
            train_loss += loss.item() * n_graphs
            epoch_n_graphs += n_graphs
        train_loss /= epoch_n_graphs
    
        train_losses.append(train_loss)
    print(train_losses)
if __name__ == "__main__":
    main()