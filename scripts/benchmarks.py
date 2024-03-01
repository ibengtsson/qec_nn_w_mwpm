import torch
import sys

sys.path.append("../")
from src.simulations import SurfaceCodeSim
from src.models import GraphNN
from src.utils import time_it, inference


def main():

    # generate data
    reps = 5
    code_sz = 5
    p = 1e-3
    n_shots = 20000
    m_nearest_nodes = 20

    sim = SurfaceCodeSim(reps, code_sz, p, n_shots)
    syndromes, flips, _ = sim.generate_syndromes(use_for_mwpm=True)

    # create a network to get realistic output shapes
    model = GraphNN()
    model.eval()
    
    # run benchmarks 
    n_its = 10
    print("Not using pool of workers:")
    time_it(inference, n_its, model, syndromes, flips, "z", m_nearest_nodes, torch.device("cpu"), False)
    
    print("Using pool of workers:")
    time_it(inference, n_its, model, syndromes, flips, "z", m_nearest_nodes, torch.device("cpu"), True)

if __name__ == "__main__":
    main()
