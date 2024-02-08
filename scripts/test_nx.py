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


if __name__ == "__main__":
    nx_mwpm()

