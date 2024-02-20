from typing import Any
import torch
import torch.nn as nn
import torch_geometric.nn as nng
from scipy.spatial.distance import cdist
import numpy as np
import pymatching
from qecsim.graphtools import mwpm
from src.graph import extract_edges


def mwpm_prediction(edges, weights, classes):

    # convert edges to dict
    classes = (classes > 0).astype(np.int32)
    edges_w_weights = {tuple(sorted(x)): w for x, w in zip(edges.T, weights)}
    edges_w_classes = {tuple(sorted(x)): c for x, c in zip(edges.T, classes)}
    matched_edges = mwpm(edges_w_weights)

    # need to make sure matched_edges is sorted
    matched_edges = [tuple(sorted((x[0], x[1]))) for x in matched_edges]

    # REMOVE IF WHEN WE HAVE ENSURED THAT THERE IS ALWAYS AN EVEN NUMBER OF EDGES
    if matched_edges:
        classes = np.array([edges_w_classes[edge] for edge in matched_edges])
        return classes.sum() & 1
    else:
        return 0


class MWPMLoss(torch.autograd.Function):

    # experiment will be a 1-d array of same length as syndromes, indicating whether its a memory x or memory z-exp
    @staticmethod
    def forward(
        ctx,
        edge_indx: torch.Tensor,
        edge_weights: torch.Tensor,
        edge_classes: torch.Tensor,
        batch_labels: torch.Tensor,
        labels: np.ndarray,
        factor=1.5,
    ):

        edge_attr = torch.stack([edge_weights, edge_classes], dim=1)

        # split edges and edge weights per syndrome
        (
            edges_p_graph,
            weights_p_graph,
            classes_p_graph,
            edge_map_p_graph,
        ) = extract_edges(
            edge_indx,
            edge_attr,
            batch_labels,
        )

        # we must loop through every graph since each one will have given a new set of edge weights
        preds = []
        grad_data = torch.zeros_like(edge_attr)
        grad_help = torch.zeros_like(edge_attr)

        for i, (edges, weights, classes, edge_map) in enumerate(
            zip(edges_p_graph, weights_p_graph, classes_p_graph, edge_map_p_graph)
        ):
            edges = edges.cpu().numpy()
            weights = weights.cpu().numpy()
            classes = classes.cpu().numpy()

            prediction = mwpm_prediction(edges, weights, classes)
            preds.append(prediction)

            # we need a workaround for gradient computations
            preds_partial_de = []
            for j in range(edges.shape[1]):
                _weights = weights.copy()

                _weights[j] = _weights[j] * factor
                delta = _weights[j] - weights[j]
                pred_w = mwpm_prediction(edges, _weights, classes)
                preds_partial_de.append([pred_w, delta])

            # REMOVE WHEN WE KNOW THAT ALL SYNDROMES HAVE AN EDGE
            if edge_map.numel() == 0:
                continue
            else:
                grad_data[edge_map, :] = torch.tensor(
                    preds_partial_de, dtype=torch.float32
                )
                grad_help[edge_map, 0] = prediction
                grad_help[edge_map, 1] = labels[i]
        preds = np.array(preds)

        # compute accuracy
        n_correct = (preds == labels).sum()
        accuracy = n_correct / labels.shape[0]
        loss = torch.tensor(1 - accuracy, requires_grad=True)

        ctx.save_for_backward(
            grad_data,
            grad_help,
        )

        return loss

    @staticmethod
    def backward(
        ctx,
        grad_output,
    ):
        grad_data, grad_help = ctx.saved_tensors
        shift_preds = grad_data[:, 0]
        delta = grad_data[:, 1]

        preds = grad_help[:, 0]
        labels = grad_help[:, 1]

        gradients = (
            (0.5 * (shift_preds + preds) - labels) * (shift_preds - preds) / delta
        )

        gradients.requires_grad = True

        return None, gradients, None, None, None, None, None


class SplitSyndromes(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, edges, edge_attr, detector_labels):
        
        node_range = torch.arange(0, detector_labels.shape[0]).to(edges.device)
        node_subset = node_range[detector_labels]

        valid_labels = torch.isin(edges, node_subset).sum(dim=0) == 2
        edges = edges[:, valid_labels]
        edge_attr = edge_attr[valid_labels, :]

        return edges, edge_attr


class GraphNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.gc = nng.GraphConv(5, 16)
        self.split_syndromes = SplitSyndromes()

        self.lin1 = nn.Linear(33, 16)
        self.lin2 = nn.Linear(16, 1)

    def forward(
        self,
        x,
        edges,
        edge_attr,
        detector_labels,
        warmup=False,
    ):

        x = self.gc(x, edges, edge_attr[:, 0] * edge_attr[:, 1])
        x = torch.nn.functional.tanh(x)

        edges, edge_attr = self.split_syndromes(edges, edge_attr, detector_labels)
        x_src, x_dst = x[edges[0, :]], x[edges[1, :]]
        edge_feat = torch.cat([x_src, edge_attr[:, [0]], x_dst], dim=-1)
        edge_feat = self.lin1(edge_feat)
        edge_feat = torch.nn.functional.tanh(edge_feat)
        edge_feat = self.lin2(edge_feat)

        if warmup:
            label = edge_attr[:, [0]]
            return edge_feat, label
        else:
            # save edges with largest weights
            n_edges = edge_feat.shape[0]
            edge_feat = edge_feat.reshape(-1, n_edges // 2)
            edge_classes = edge_attr[:, 1].reshape(-1, n_edges // 2)
            max_inds = torch.argmin(edge_feat, dim=0)

            edge_feat = edge_feat[max_inds, range(n_edges // 2)]
            edge_classes = edge_classes[max_inds, range(n_edges // 2)]

            edges = edges[:, : n_edges // 2]

            return edges, edge_feat, edge_classes
