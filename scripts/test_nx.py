import torch
import torch.nn as nn
import torch_geometric.nn as nng
from scipy.spatial.distance import cdist
import stim
import numpy as np
import pymatching as pm
from torch_geometric.data import Data


import sys

sys.path.append("../")
from src.simulations import SurfaceCodeSim
from src.graph import get_batch_of_graphs, get_3D_graph
import networkx as nx
import torch_geometric.utils as tu


def nx_mwpm():
    reps = 1
    code_sz = 3
    p = 1e-1
    n_shots = 1
    sim = SurfaceCodeSim(
        reps, code_sz, p, n_shots, code_task="surface_code:rotated_memory_z"
    )
    syndromes, flips, _ = sim.generate_syndromes(n_syndromes = 1, n_shots = n_shots)
    x, edge_index, edge_attr, batch_labels = get_batch_of_graphs(
        syndromes, m_nearest_nodes=5)

    data = Data(x, edge_index, edge_attr)
    nx_graph = tu.to_networkx(data, to_undirected = True)
    m = pm.Matching.from_networkx(nx_graph)
    num_det = m.num_detectors
    z = np.ones(num_det)
    print(z)
    import matplotlib.pyplot as plt
    #m.draw()
    #plt.show()
    print(m)
    corr, weights = m.decode(z, return_weight = True)
    print(corr)
    print(weights)
    return m


def reshape_edges(edges, edge_weights, batch_labels, n_nodes):
    node_range = torch.arange(0, n_nodes)

    edges_per_syndrome = []
    weights_per_syndrome = []
    for i in range(batch_labels[-1] + 1):
        ind_range = torch.nonzero(batch_labels == i)
        edge_mask = (edges >= node_range[ind_range[0]]) & (
            edges <= node_range[ind_range[-1]]
        )
        new_edges = edges[:, edge_mask[0, :]]
        new_weights = edge_weights[edge_mask[0, :]]

        edges_per_syndrome.append(new_edges)
        weights_per_syndrome.append(new_weights)

    return edges_per_syndrome, weights_per_syndrome



def graphs():

    reps = 2
    code_sz = 3
    p = 1e-3
    n_shots = 100
    sim = SurfaceCodeSim(
        reps, code_sz, p, n_shots, code_task="surface_code:rotated_memory_z"
    )
    syndromes, flips, _ = sim.generate_syndromes(n_shots)

    x, edge_index, edge_attr, batch_labels = get_batch_of_graphs(
        syndromes, m_nearest_nodes=5
    )
    print(sim.detector_indx)
    
    
    edges_per_syndrome, weights_per_syndrome = reshape_edges(edge_index, edge_attr, batch_labels, x.shape[0])

if __name__ == "__main__":
    #stim_mwpm()
    # main()
    # test_nn()
    #graphs()
    nx_mwpm()

