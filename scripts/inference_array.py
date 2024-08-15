import argparse
import torch
import sys
import csv

sys.path.append("..")

from pathlib import Path
from src.models import SimpleGraphNNV4, GraphAttentionV3
from src.simulations import SurfaceCodeSim
from src.utils import inference, attention_inference


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
        data = torch.load(model_path, map_location=device)
    else:
        raise FileNotFoundError("The file was not found!")

    # determine what model we should load state_dict into and save model information
    state_dict = data["model"]
    gcn_layers = data["model_settings"]["hidden_channels_GCN"]
    gcn_dims = "".join(
        [str(dim) + " " for dim in gcn_layers[:-1]] + [str(gcn_layers[-1])]
    )
    mixed = (
        "Constant"
        if data["graph_settings"]["min_error_rate"]
        == data["graph_settings"]["max_error_rate"]
        else "Mixed"
    )

    if "attention.qkv_proj.weight" in state_dict.keys():
        n_heads = 3 if "3 heads" in data["training_settings"]["comment"] else 1
        model = GraphAttentionV3(
            hidden_channels_GCN=gcn_layers,
            num_heads=n_heads,
        ).to(device)
        inference_function = attention_inference
        name = "Attention"
        mlp_dims = f"Follows from GCN-dim - {n_heads} head(s)"

    else:
        mlp_layers = data["model_settings"]["hidden_channels_MLP"]
        model = SimpleGraphNNV4(
            hidden_channels_GCN=gcn_layers,
            hidden_channels_MLP=mlp_layers,
        ).to(device)
        inference_function = inference
        name = "Feed-forward"
        mlp_dims = "".join(
            [str(dim) + " " for dim in mlp_layers[:-1]] + [str(mlp_layers[-1])]
        )

    # compute number of parameters in model
    tot_params = sum(p.numel() for p in model.parameters())

    # load weights and biases
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Moved model to {device} and loaded pre-trained weights.")

    # settings
    n_graphs = int(1e7)
    n_graphs_per_sim = int(5e4)
    p = 1e-3

    m_nearest_nodes = data["graph_settings"]["m_nearest_nodes"]

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
    code_task = "surface_code:rotated_memory_z"
    experiment = code_task[-1]
    sim = SurfaceCodeSim(
        reps,
        code_sz,
        p,
        n_shots=n_graphs_per_sim,
        code_task=code_task,
    )

    # go through partitions
    correct_preds = 0
    n_trivial = 0
    for _ in range(n_partitions):
        syndromes, flips, n_identities = sim.generate_syndromes(use_for_mwpm=True)

        # add identities to # trivial predictions
        n_trivial += n_identities

        # run inference
        _correct_preds = inference_function(
            model,
            syndromes,
            flips,
            m_nearest_nodes=m_nearest_nodes,
            device=device,
            experiment=experiment,
        )
        correct_preds += _correct_preds

    # run the remaining graphs
    if remaining > 0:

        syndromes, flips, n_identities = sim.generate_syndromes(use_for_mwpm=True)

        # add identities to # trivial predictions
        n_trivial += n_identities

        _correct_preds = inference_function(
            model,
            syndromes,
            flips,
            m_nearest_nodes=m_nearest_nodes,
            device=device,
            experiment=experiment,
        )
        correct_preds += _correct_preds

    # compute logical failure rate
    failure_rate = (n_graphs - correct_preds - n_trivial) / n_graphs
    print(f"We have a logical failure rate of {failure_rate}.")

    # append results to file
    file_path = Path("../data/failure_rates_updated.csv")
    data = [model_path.name, name, gcn_dims, mlp_dims, mixed, tot_params, failure_rate]

    if file_path.is_file():
        with open(file_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(data)
    else:
        with open(file_path, "x", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Path",
                    "Type of head",
                    "GCN-layers",
                    "MLP-layers",
                    "Error rates",
                    "# parameters",
                    "Failure rate",
                ]
            )
            writer.writerow(data)

    return 0


if __name__ == "__main__":
    main()
