import torch
import torch.nn as nn
import torch_geometric.nn as nng
import numpy as np
import sys
sys.path.append("../")
from src.simulations import SurfaceCodeSim
from src.graph import get_batch_of_graphs
from src.models import MWPMLoss
from src.utils import inference


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
    
class LocalSearch:
    def __init__(self, model):
        self.model = model
        self.initial_score = torch.tensor(float(0))
        self.top_score = self.initial_score
        self.target = None
        self.vector = torch.nn.utils.parameters_to_vector(model.parameters())
        self.elite = self.vector.clone()
        self.n = self.vector.numel()
        self.running_idxs = np.arange(self.n)
        np.random.shuffle(self.running_idxs)
        self.idx = 0
        self.value = []  # list of indices
        self.num_selections = 25
        self.magnitude = torch.empty(self.num_selections,
                                dtype=self.vector.dtype,
                                device=self.vector.device)
        self.noise_vector = torch.empty_like(self.vector)
        self.jumped = False

    def set_value(self):
        """Use the numpy choices function (which has no equivalent in Pytorch)
        to generate a sample from the array of indices. The sample size and
        distribution are dynamically updated by the algorithm's state.
        """
        self.check_idx()
        choices = self.running_idxs[self.idx:self.idx+self.num_selections]
        self.value = choices
        self.idx+=self.num_selections

    def check_idx(self):
        if (self.idx+self.num_selections)>self.n:
            self.idx=0
            np.random.shuffle(self.running_idxs)

    def set_noise(self):
        # Cast to precision and CUDA, and edit shape
        # 0.5 can be adjusted to fit scale of noise
        self.magnitude.uniform_(-0.5, 0.5).squeeze()

    def set_noise_vector(self):
        """ This function defines a noise tensor, and returns it. The noise
        tensor needs to be the same shape as our originial vecotr. Hence, a
        "basis" tensor is created with zeros, then the chosen indices are
        modified.
        """
        self.noise_vector.fill_(0.)
        self.noise_vector[self.value] = self.magnitude

    def update_weights(self, model):
        nn.utils.vector_to_parameters(self.vector, model.parameters())

    def set_elite(self):
        self.jumped = False
        self.elite[self.value] = self.vector[self.value]
            #self.elite.clamp_(-0.9, 0.9)
            #self.elite.copy_(self.vector)
        self.jumped = True
            #self.frustration.reset_state()

    def step(self,syndrome,flips):
        accuracy = inference(self.model,syndrome,flips)
        for i in range(0,10):
            self.set_value()
            self.set_noise()
            self.set_noise_vector()
            self.vector[torch.from_numpy(self.value)] = self.vector[torch.from_numpy(self.value)] + torch.from_numpy(self.value)
            self.update_weights(self.model)
            new_accuracy = inference(self.model,syndrome,flips)
            if new_accuracy > accuracy:
                self.set_elite()
                self.top_score = new_accuracy
            self.idx += 1

    def return_topscore(self):
        return self.top_score
            


def main():
    
    reps = 3
    code_sz = 3
    p = 1e-3
    n_shots = 8000
    sim = SurfaceCodeSim(reps, code_sz, p, n_shots)
    n_epochs = 10
    n_batches = 5
    factor = 0.5
    
    model = GraphNN()
    model.train()
    # loss_fun = nn.MSELoss()
    #loss_fun = MWPMLoss.apply
    optim = LocalSearch(model)
    
    for epoch in range(n_epochs):
        train_acc = 0
        epoch_n_graphs = 0
        
        # if epoch > 0:
        #     params = list(model.parameters())
        #     for i, p in enumerate(params):
        #         print(f"Parameter {i}: {torch.count_nonzero(p.grad).item()} non-zero gradients.")

        for i in range(n_batches):
            #optim.zero_grad()
            syndromes, flips, n_trivial = sim.generate_syndromes(n_shots)
            #x, edges, edge_attr, batch_labels, detector_labels = get_batch_of_graphs(syndromes, 10)
            #edges, edge_feat = model(x, edges, edge_attr, detector_labels)
            optim.step(syndromes,flips)
            #node_range = torch.arange(0, x.shape[0])
            # loss = loss_fun(
            #     edges,
            #     edge_feat,
            #     batch_labels,
            #     node_range,
            #     np.array(flips) * 1,
            #     factor
            #     )
    
            #loss.backward()
            #optim.step()
            n_graphs = syndromes.shape[0]
            top_accuracy = optim.return_topscore()
            train_acc += top_accuracy * n_graphs
            #train_loss += loss.item() * n_graphs
            epoch_n_graphs += n_graphs
            print(i)
        train_acc /= epoch_n_graphs
    
        print(train_acc)

if __name__ == "__main__":
    main()