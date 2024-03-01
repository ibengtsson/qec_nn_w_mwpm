import torch
import torch.nn as nn
import torch_geometric.nn as nng
from scipy.spatial.distance import cdist
import stim
import numpy as np
import pymatching as pm
import matplotlib.pyplot as plt
from torch_geometric.utils import coalesce, is_undirected, sort_edge_index

import sys
sys.path.append("../")
from src.simulations import SurfaceCodeSim
from src.graph import get_batch_of_graphs, extract_edges, extract_edges_v2
from src.utils import time_it

def main():

    reps = 5
    code_sz = 5
    p = 1e-3
    n_shots = 5000

    sim = SurfaceCodeSim(reps, code_sz, p, n_shots)
    syndromes, flips, _ = sim.generate_syndromes(use_for_mwpm=True)
    # x, edges, edge_attr, batch_labels, detector_labels = get_batch_of_graphs(syndromes, 20)
    
    time_it(get_batch_of_graphs, 10, syndromes, 20)

    return 

if __name__ == "__main__":
    main()
