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
from src.graph import get_batch_of_graphs, extract_edges
from src.utils import time_it
from src.models import GraphNN, GraphNNV2
from qecsim.graphtools import mwpm
import logging
logging.disable(sys.maxsize)


def main():

    edges = {(0, 1): 0.1, (0, 2): 0, (0, 3): 0, (1, 2): 0, (2, 3): 0, (5, 7): 2, (7, 8): 1, (5, 8): 1, (7, 8): 2, (6, 7): 2}
    match_edges = mwpm(edges)

    print(match_edges)
    return 

if __name__ == "__main__":
    main()
