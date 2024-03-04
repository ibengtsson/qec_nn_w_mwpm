import datetime
import yaml
import torch
import torch.nn as nn
import numpy as np
from src.graph import get_batch_of_graphs, extract_edges
from src.models import mwpm_prediction
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, cpu_count

def time_it(func, reps, *args):
    start_t = datetime.datetime.now()
    for i in range(reps):
        func(*args)
    t_per_loop = (datetime.datetime.now() - start_t) / reps
    print(t_per_loop)


def parse_yaml(yaml_config):

    if yaml_config is not None:
        with open(yaml_config, "r") as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    # default settings
    else:
        config = {}
        config["paths"] = {
            "root": "../",
            "save_dir": "../training_outputs",
            "saved_model_path": "..",
        }
        config["graph_settings"] = {
            "experiment": "z",
            "code_size": 5,
            "repetitions": 5,
            "min_error_rate": 0.001,
            "max_error_rate": 0.005,
            "m_nearest_nodes": 10,
            "n_node_features": 5,
            "power": 2,
            "n_classes": 1,
        }
        config["model_settings"] = {
            "hidden_channels_GCN": [32, 64, 128, 256, 512],
            "hidden_channels_MLP": [256, 128, 64],
        }
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config["training_settings"] = {
            "seed": None,
            "dataset_size": int(1e4),
            "validation_set_size": int(1e3),
            "batch_size": int(5e3),
            "warmup_epochs": 100,
            "tot_epochs": 10,
            "gradient_factor": 3,
            "warmup_lr": 0.01,
            "lr": 0.001,
            "device": device,
            "resume_training": False,
            "current_epoch": 0,
        }

    # read settings into variables
    paths = config["paths"]
    graph_settings = config["graph_settings"]
    model_settings = config["model_settings"]
    training_settings = config["training_settings"]

    return paths, graph_settings, model_settings, training_settings


def inference(
    model: nn.Module,
    syndromes: np.ndarray,
    flips: np.ndarray,
    experiment: str = "z",
    m_nearest_nodes: int = 10,
    device: torch.device = torch.device("cpu"),
    pool: bool = False,
):

    x, edge_index, edge_attr, batch_labels, detector_labels = get_batch_of_graphs(
        syndromes, m_nearest_nodes, experiment=experiment, device=device
    )
    edge_index, edge_weights, edge_classes = model(
        x, edge_index, edge_attr, detector_labels, batch_labels,
    )

    if pool:
        preds = predict_mwpm_with_pool(
            edge_index, edge_weights, edge_classes, batch_labels
        )
    else:
        preds = predict_mwpm(edge_index, edge_weights, edge_classes, batch_labels)

    n_correct = (preds == flips).sum()
    accuracy = n_correct / len(preds)
    return n_correct, accuracy


def predict_mwpm(
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    edge_classes: torch.Tensor,
    batch_labels: torch.Tensor,
):
    # split edges and edge weights per syndrome
    edge_attr = torch.stack([edge_weights, edge_classes], dim=1)
    edges_p_graph, weights_p_graph, classes_p_graph, _ = extract_edges(
        edge_index,
        edge_attr,
        batch_labels,
    )

    preds = []
    for edges, weights, classes in zip(edges_p_graph, weights_p_graph, classes_p_graph):
        edges = edges.cpu().numpy()
        weights = weights.detach().cpu().numpy()
        classes = classes.detach().cpu().numpy()
        p = mwpm_prediction(edges, weights, classes)
        preds.append(p)

    return np.array(preds)


# ctrl+c termination should be supported now, but use this function with some caution!
def predict_mwpm_with_pool(
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    edge_classes: torch.Tensor,
    batch_labels: torch.Tensor,
):
    # split edges and edge weights per syndrome

    edge_attr = torch.stack([edge_weights, edge_classes], dim=1)
    edges_p_graph, weights_p_graph, classes_p_graph, _ = extract_edges(
        edge_index,
        edge_attr,
        batch_labels,
    )

    preds = []
    edges_p_graph = [t.cpu().numpy() for t in edges_p_graph]
    weights_p_graph = [t.cpu().detach().numpy() for t in weights_p_graph]
    classes_p_graph = [t.cpu().detach().numpy() for t in classes_p_graph]
    chunk_size = 500
    with Pool(processes=(cpu_count() - 1)) as p:
        preds = p.starmap(
            mwpm_prediction,
            list(zip(edges_p_graph, weights_p_graph, classes_p_graph)),
            chunksize=chunk_size,
        )

    return np.array(preds)