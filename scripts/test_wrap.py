import torch
import torch.nn as nn
import torch_geometric.nn as nng
from scipy.spatial.distance import cdist
import stim
import numpy as np
import pymatching as pm
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph


import sys

sys.path.append("../")
from src.simulations import SurfaceCodeSim
from src.graph import get_batch_of_graphs, get_3D_graph
import networkx as nx
import torch_geometric.utils as tu

def cylinder_distance(x, y, width, wrap_axis=1, manhattan=False):
    # x, y have coordinates (x, y, t)
    ds = np.abs(x - y)
    eq_class = ds[:, wrap_axis] > 0.5 * width
    ds[eq_class, wrap_axis] = width - ds[eq_class, wrap_axis]

    if not manhattan:
        return np.sqrt((ds ** 2).sum(axis=1)), eq_class
    else:
        return ds.sum(axis=1), eq_class
    
p1s = np.array([
    [1, 2, 3, 4],
    [7, 2, 9, 2]
])

p2s = np.array([
    [3, 4, 9, 0],
    [7, 3, 2, 1]
])

distances, eq_class = cylinder_distance(p1s.T, p2s.T, 10, manhattan=False)

def generate_graphs():
    reps = 1
    code_sz = 3
    p = 1e-1
    n_shots = 100
    sim = SurfaceCodeSim(
        reps, code_sz, p, n_shots, code_task="surface_code:rotated_memory_z"
    )
    syndromes, flips, _ = sim.generate_syndromes(n_shots)
    
    x, edge_index, edge_attr, batch_labels = get_batch_of_graphs2(
        syndromes, m_nearest_nodes=5, code_distance=code_sz)
    


def get_batch_of_graphs2(
    syndromes,
    m_nearest_nodes,
    code_distance,
    experiment="z",
    n_node_features=5,
    power=2.0,
    device=torch.device("cpu"),
):
    syndromes = syndromes.astype(np.float32)
    defect_inds = np.nonzero(syndromes)
    defects = syndromes[defect_inds]

    defect_inds = np.transpose(np.array(defect_inds))
    _, defects_per_graph = np.unique(defect_inds[:,0], return_counts=True)
    # this function currently handles x OR z errors only
    print(defects_per_graph)
    graphs_with_boundary_node = defects_per_graph % 2 != 0
    print(graphs_with_boundary_node)

    x_defects = defects == 1
    z_defects = defects == 3

    node_features = np.zeros((defects.shape[0], n_node_features + 1), dtype=np.float32)

    node_features[x_defects, 0] = 1
    node_features[x_defects, 2:] = defect_inds[x_defects, ...]
    node_features[z_defects, 1] = 1
    node_features[z_defects, 2:] = defect_inds[z_defects, ...]

    node_features.max(axis=0)
    x_cols = [0, 1, 3, 4, 5]
    batch_col = 2

    x = torch.tensor(node_features[:, x_cols]).to(device)
    batch_labels = torch.tensor(node_features[:, batch_col]).long().to(device)

    pos = x[:, 2:]

    # get edge indices
    edge_index = knn_graph(pos, m_nearest_nodes, batch=batch_labels)
    # compute distances to get edge attributes
    width = code_distance + 1
    if experiment == "z":
        wrap_axis = 1
    else:
        wrap_axis = 0
    dist, eq_class = cylinder_distance(pos[edge_index[0], :], pos[edge_index[1], :], width, wrap_axis=wrap_axis)
    
    # cast eq_class to float32
    eq_class = eq_class.type(dtype=torch.float32)
    edge_attr = torch.stack((dist**power, eq_class), dim=1)

    return x, edge_index, edge_attr, batch_labels
