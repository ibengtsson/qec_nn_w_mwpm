import argparse
import torch
import sys

sys.path.append("../")
from src.models import GraphNN, SimpleGraphNNV5
from src.simulations import SurfaceCodeSim
from src.utils import inference, inference_TEST

from pathlib import Path


def main():

    # command line parsing
    parser = argparse.ArgumentParser(description="Choose model to load.")
    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-d", "--device", required=False)

    args = parser.parse_args()

    model_path = Path(args.file)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    if model_path.is_file():
        model_data = torch.load(model_path, map_location=device)
    else:
        raise FileNotFoundError("The file was not found!")

    gcn_layers = model_data["model_settings"]["hidden_channels_GCN"]
    mlp_layers = model_data["model_settings"]["hidden_channels_MLP"]

    # model = GraphNN(
    #     hidden_channels_GCN=gcn_layers, hidden_channels_MLP=mlp_layers
    # ).to(device)
    model = SimpleGraphNNV5(
        hidden_channels_GCN=gcn_layers,
        hidden_channels_MLP=mlp_layers,
        ).to(device)
    model.load_state_dict(model_data["model"])
    model.eval()

    print(f"Moved model to {device} and loaded pre-trained weights.")

    # settings
    n_graphs = int(1e6)
    n_graphs_per_sim = int(1e5)
    p = 1e-3

    m_nearest_nodes = model_data["graph_settings"]["m_nearest_nodes"]

    # if we want to run inference on many graphs, do so in batches
    if n_graphs > n_graphs_per_sim:
        n_partitions = n_graphs // n_graphs_per_sim
        remaining = n_graphs % n_graphs_per_sim
    else:
        n_partitions = 0
        remaining = n_graphs

    # read code distance and number of repetitions from file name
    file_name = model_path.name
    splits = file_name.split("_")
    code_sz = int(splits[0][1]) 
    reps = int(splits[3].split(".")[0])
    
    # initialise simulation
    sim = SurfaceCodeSim(
            reps,
            code_sz,
            p,
            n_shots=n_graphs_per_sim,
        )
    
    # go through partitions
    correct_preds = 0
    n_trivial = 0
    for _ in range(n_partitions):
        syndromes, flips, n_identities = sim.generate_syndromes(use_for_mwpm=True)

        # add identities to # trivial predictions
        n_trivial += n_identities

        # run inference
        _correct_preds = inference(
            model, syndromes, flips, m_nearest_nodes=m_nearest_nodes, device=device
        )
        correct_preds += _correct_preds

    # run the remaining graphs
    if remaining > 0:

        syndromes, flips, n_identities = sim.generate_syndromes(use_for_mwpm=True)

        # add identities to # trivial predictions
        n_trivial += n_identities

        _correct_preds = inference(
            model, syndromes, flips, m_nearest_nodes=m_nearest_nodes, device=device
        )
        correct_preds += _correct_preds

    # compute logical failure rate
    failure_rate = (n_graphs - correct_preds - n_trivial) / n_graphs
    print(f"We have a logical failure rate of {failure_rate}.")

    return 0


if __name__ == "__main__":
    main()
