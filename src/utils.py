import datetime
import yaml
import torch
import torch.nn as nn
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.ticker as mticker
#import seaborn as sns
from qecsim.graphtools import mwpm
from src.graph import get_batch_of_graphs, extract_edges
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, cpu_count
import pandas as pd
import sys
import logging
logging.disable(sys.maxsize)
from torch_geometric.utils import to_dense_adj
import random

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
            "tot_epochs": 10,
            "lr": 0.001,
            "device": device,
            "resume_training": False,
            "current_epoch": 0,
            "comment": "No comment"
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
    nested_tensors: bool = False,
):
    # set model in inference mode
    # model.eval()
    x, edge_index, edge_attr, batch_labels, detector_labels = get_batch_of_graphs(
        syndromes, m_nearest_nodes, experiment=experiment, device=device
    )
    edge_index, edge_weights, edge_classes = model(
        x, edge_index, edge_attr, detector_labels, batch_labels,
    )
    
    edge_weights = torch.nn.functional.sigmoid(edge_weights)
    if nested_tensors:
        preds = predict_mwpm_nested(edge_index, edge_weights, edge_classes)
    else:
        preds = predict_mwpm(edge_index, edge_weights, edge_classes, batch_labels)

    #correct_or_not = (preds == flips)
    #wrong_syndromes = syndromes[~correct_or_not, ...]
    #wrong_flips = flips[~correct_or_not]
    #inds = random.sample(range(wrong_syndromes.shape[0]), 4)
    #samples = wrong_syndromes[inds, ...]
    #sample_flips = wrong_flips[inds]
    #for s, f in zip(samples, sample_flips):
    #    plot_syndrome(s, f)
    n_correct = (preds == flips).sum()
    
    return n_correct

def inference_TEST(
    model: nn.Module,
    syndromes: np.ndarray,
    flips: np.ndarray,
    experiment: str = "z",
    m_nearest_nodes: int = 10,
    device: torch.device = torch.device("cpu"),
    nested_tensors: bool = False,
):
    # set model in inference mode
    model.eval()
    x, edge_index, edge_attr, batch_labels, detector_labels = get_batch_of_graphs(
        syndromes, m_nearest_nodes, experiment=experiment, device=device
    )
    edge_index, edge_weights, edge_classes = model(
        x, edge_index, edge_attr, detector_labels,
    )

    if nested_tensors:
        preds = predict_mwpm_nested(edge_index, edge_weights, edge_classes)
    else:
        preds = predict_mwpm(edge_index, edge_weights, edge_classes, batch_labels)

    n_correct = (preds == flips).sum()

    # # confusion plot
    # true_identity = ((preds == 0) & (flips == 0)).sum() / (flips == 0).sum() * 100
    # true_flip = ((preds == 1) & (flips == 1)).sum() / (flips == 1).sum() * 100
    # false_identity = ((preds == 0) & (flips == 1)).sum() / (flips == 1).sum() * 100
    # false_flip = ((preds == 1) & (flips == 0)).sum() / (flips == 0).sum() * 100

    # confusion_data = [[true_identity, false_identity], [false_flip, true_flip]]
    # df_confusion = pd.DataFrame(
    #     confusion_data,
    #     index=["Predicted 0 (%)", "Predicted 1 (%)"],
    #     columns=["True 0", "True 1"],
    # )
    # pd.set_option("display.precision", 2)
    # print(df_confusion)
    return n_correct

def attention_inference(
    model: nn.Module,
    syndromes: np.ndarray,
    flips: np.ndarray,
    experiment: str = "z",
    m_nearest_nodes: int = 10,
    n: int = 8,
    device: torch.device = torch.device("cpu"),
):
    # set model in inference mode
    model.eval()
    
    # run chunks
    stabilizer_label = {"z": 3, "x": 1}
    syndrome_chunks, flip_chunks = split_syndromes_equisize(syndromes, flips, n, stabilizer_label[experiment])
    
    n_correct = 0
    for s, f in zip(syndrome_chunks, flip_chunks):
    
        x, edge_index, edge_attr, batch_labels, detector_labels = get_batch_of_graphs(
            s, m_nearest_nodes, experiment=experiment, device=device
        )
        edge_index, edge_weights, edge_classes = model(
            x, edge_index, edge_attr, detector_labels, batch_labels,
        )
        
        edge_weights = torch.nn.functional.sigmoid(edge_weights)
        preds = predict_mwpm_attention(edge_index, edge_weights, edge_classes)

        n_correct += (preds == f).sum()

    return n_correct


def predict_mwpm_attention(
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    edge_classes: torch.Tensor,
):
    preds = []
    for edges, weights, classes in zip(edge_index, edge_weights, edge_classes):

        # begin by removing the trailing zeros from the padding
        mask = edges[:, 0] != edges[:, 1]
        edges = edges[mask, :].T
        weights = weights[mask]
        classes = classes[mask]

        # run MWPM
        edges = edges.cpu().numpy()
        weights = weights.detach().cpu().numpy().squeeze()
        classes = classes.detach().cpu().numpy().squeeze()
        p = mwpm_prediction(edges, weights, classes)
        preds.append(p)

    return np.array(preds)

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

def predict_mwpm_nested(edge_index, edge_weights, edge_classes):

    preds = []
    for edges, weights, classes in zip(edge_index, edge_weights, edge_classes):

        # find which (of the two) egdes that have the minimum weight for each node pair
        weights = torch.cat([weights[::2], weights[1::2]], dim=1)
        classes = torch.cat([classes[::2], classes[1::2]], dim=1)
        edges = edges[:, ::2]

        weights, min_inds = torch.min(weights, dim=1)
        classes = classes[range(edges.shape[1]), min_inds]

        # do a softmax on the weights
        weights = torch.nn.functional.softmax(weights, dim=0)

        # move to CPU and run MWPM
        edges = edges.detach().cpu().numpy()
        weights = weights.detach().cpu().numpy()
        classes = classes.detach().cpu().numpy()
        p = mwpm_prediction(edges, weights, classes)
        preds.append(p)

    return np.array(preds)

def mwpm_prediction(edges, weights, classes):

    classes = classes.astype(np.int32)

    # if only one edge, we only have one matching
    if edges.shape[1] == 1:
        flip = classes.sum() & 1
        return flip

    edges_w_weights = {tuple(sorted(x)): w for x, w in zip(edges.T, weights)}
    edges_w_classes = {tuple(sorted(x)): c for x, c in zip(edges.T, classes)}

    matched_edges = mwpm(edges_w_weights)

    # need to make sure matched_edges is sorted
    matched_edges = [tuple(sorted((x[0], x[1]))) for x in matched_edges]
    classes = np.array([edges_w_classes[edge] for edge in matched_edges])
    flip = classes.sum() & 1

    return flip


def mwpm_w_grad(edges, weights, classes):

    classes = (classes > 0).astype(np.int32)
    # if only one edge, we only have one matching
    if edges.shape[1] == 1:
        flip = classes.sum() & 1
        gradient = torch.ones(weights.shape)
        return flip, gradient

    edges_w_weights = {tuple(sorted(x)): w for x, w in zip(edges.T, weights)}
    edges_w_classes = {tuple(sorted(x)): c for x, c in zip(edges.T, classes)}
    edge_range = {tuple(sorted(x)): i for i, x in enumerate(edges.T)}

    matched_edges = mwpm(edges_w_weights)

    # need to make sure matched_edges is sorted
    matched_edges = [tuple(sorted((x[0], x[1]))) for x in matched_edges]

    classes = np.array([edges_w_classes[edge] for edge in matched_edges])
    flip = classes.sum() & 1
    match_inds = [edge_range[edge] for edge in matched_edges]
    mask = np.zeros(weights.shape, dtype=bool)
    mask[match_inds] = True

    gradient = torch.zeros(weights.shape)
    gradient[mask] = 1

    return flip, gradient

def mwpm_w_grad_v2(edges, weights, classes):

    if np.unique(edges).shape[0] & 1:
        print("Warning!")
        print(edges)

    classes = classes.astype(np.int32)

    # if only one edge, we only have one matching
    if edges.shape[1] == 1:
        flip = classes.sum() & 1
        mask = np.ones(shape=(1, 1), dtype=bool)

        return flip, mask


    edges_w_weights = {tuple(sorted(x)): w for x, w in zip(edges.T, weights)}
    edges_w_classes = {tuple(sorted(x)): c for x, c in zip(edges.T, classes)}
    edge_range = {tuple(sorted(x)): i for i, x in enumerate(edges.T)}

    matched_edges = mwpm(edges_w_weights)


    # need to make sure matched_edges is sorted
    matched_edges = [tuple(sorted((x[0], x[1]))) for x in matched_edges]

    classes = np.array([edges_w_classes[edge] for edge in matched_edges])
    flip = classes.sum() & 1
    match_inds = [edge_range[edge] for edge in matched_edges]
    mask = np.zeros(weights.shape, dtype=bool)
    mask[match_inds] = True

    return flip, mask

def split_syndromes_equisize(syndromes, flips, n, exp_label=3):
    
    # count nodes per graphs and add 1 where we need to add virtual nodes
    n_nodes_p_graph = np.count_nonzero(syndromes, axis=(1, 2, 3))
    odd_n_nodes = np.count_nonzero(syndromes == exp_label, axis=(1, 2, 3)) & 1
    n_nodes_p_graph += odd_n_nodes
    
    # sort everything to make sure graphs of similar size belongs to same chunk
    sort_args = np.argsort(n_nodes_p_graph)
    syndromes = syndromes[sort_args, ...]
    flips = flips[sort_args]
    n_nodes_p_graph = n_nodes_p_graph[sort_args]
    
    # cumulative sum
    cum_sum = n_nodes_p_graph.cumsum()
    
    # approximate sum to aim for
    partsum = cum_sum[-1] // n
    
    # generates the cumulative sums of each part
    cumpartsums = np.array(range(1, n))*partsum
    
    # finds the indices where the cumulative sums are sandwiched
    inds = np.searchsorted(cum_sum, cumpartsums) 
    
    # split
    syndrome_chunks = np.split(syndromes, inds)
    flip_chunks = np.split(flips, inds)
    
    return syndrome_chunks, flip_chunks
    
def plot_syndrome(syndrome, flip):
    sz, _, reps = syndrome.shape
    n = (reps // 2) + (reps % 2)
    
    fig, axes = plt.subplots(nrows=n , ncols=2, figsize=(6, n*3))
    colors = sns.color_palette(palette="Set2", n_colors=3)
    colors = ["white", "lawngreen", "royalblue"]
    cbar_ticks = [0.5, 1.5, 2.5]
    cbar_tick_labels = mticker.FixedFormatter(["Identity", "X", "Z"])
    for i, ax in enumerate(axes.flatten()):
        
        sns.heatmap(syndrome[:, :, i], vmin=0, vmax=3, cmap=colors, square=True, ax=ax, cbar_kws={"ticks": cbar_ticks, "format": cbar_tick_labels})
        # ax.invert_yaxis()
        ax.hlines(range(1, sz), xmin=0, xmax=sz, color="k")
        ax.vlines(range(1, sz), ymin=0, ymax=sz, color="k")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t = {i}")
    
    flip_dict = {0: "No flip", 1: "Flip"}
    fig.suptitle(flip_dict[flip])
    plt.show()
