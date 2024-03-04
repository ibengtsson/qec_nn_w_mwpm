from typing import Any
import torch
import torch.nn as nn
import torch_geometric.nn as nng
from torch_geometric.utils import sort_edge_index
import numpy as np
from qecsim.graphtools import mwpm
from src.graph import extract_edges


def mwpm_prediction(edges, weights, classes):

    # convert edges to dict
    if np.unique(edges).shape[0] % 2 != 0:
        print("Odd edges")
        print(edges)
        # print(weights)
        # print(classes)
    if edges.shape[1] == 0:
        print("No edges?")
        print(edges)
        print(weights)
        print(classes)

    classes = (classes > 0).astype(np.int32)
    edges_w_weights = {tuple(sorted(x)): w for x, w in zip(edges.T, weights)}
    edges_w_classes = {tuple(sorted(x)): c for x, c in zip(edges.T, classes)}
    
    matched_edges = mwpm(edges_w_weights)

    # need to make sure matched_edges is sorted
    matched_edges = [tuple(sorted((x[0], x[1]))) for x in matched_edges]
    if matched_edges:
        classes = np.array([edges_w_classes[edge] for edge in matched_edges])
        flip = classes.sum() & 1 

        return flip
    else:
        return 0
    
def mwpm_w_grad(edges, weights, classes):
    
    classes = (classes > 0).astype(np.int32)
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
    
    gradient = torch.ones(weights.shape)
    gradient[~mask] = -1

    return flip, gradient

def mwpm_w_grad_v2(edges, weights, classes):

    classes = (classes > 0).astype(np.int32)
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
        factor: float = 2,
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
                ).to(grad_data.device)
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

        return None, gradients, None, None, None, None

class MWPMLoss_v2(torch.autograd.Function):

    # experiment will be a 1-d array of same length as syndromes, indicating whether its a memory x or memory z-exp
    @staticmethod
    def forward(
        ctx,
        edge_indx: torch.Tensor,
        edge_weights: torch.Tensor,
        edge_classes: torch.Tensor,
        batch_labels: torch.Tensor,
        labels: np.ndarray,
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
        grads = torch.zeros_like(edge_weights, device="cpu")

        for edges, weights, classes, edge_map, label in zip(edges_p_graph, weights_p_graph, classes_p_graph, edge_map_p_graph, labels):
            edges = edges.cpu().numpy()
            weights = weights.cpu().numpy()
            classes = classes.cpu().numpy()

            prediction, gradients = mwpm_w_grad(edges, weights, classes)
            preds.append(prediction)
            
            # if prediction is wrong, flip direction of gradient
            if prediction != label:
                gradients *= -1
            
            grads[edge_map] = gradients

        grads = grads.to(edge_weights.device)
        preds = np.array(preds)
        
        n_correct = (preds == labels).sum()
        accuracy = n_correct / labels.shape[0]
        loss = torch.tensor(1 - accuracy, requires_grad=True)

        ctx.save_for_backward(grads)

        return loss

    @staticmethod
    def backward(
        ctx,
        grad_output,
    ):
        grads, = ctx.saved_tensors
        grads.requires_grad = True

        return None, grads, None, None, None
    
class MWPMLoss_v3(torch.autograd.Function):

    # experiment will be a 1-d array of same length as syndromes, indicating whether its a memory x or memory z-exp
    @staticmethod
    def forward(
        ctx,
        edge_indx: torch.Tensor,
        edge_weights: torch.Tensor,
        edge_classes: torch.Tensor,
        batch_labels: torch.Tensor,
        labels: np.ndarray,
    ):

        loss_fun = torch.nn.MSELoss()
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
        desired_weights = torch.zeros_like(edge_weights, device="cpu")

        for edges, weights, classes, edge_map, label in zip(edges_p_graph, weights_p_graph, classes_p_graph, edge_map_p_graph, labels):
            edges = edges.cpu().numpy()
            weights = weights.cpu().numpy()
            classes = classes.cpu().numpy()

            prediction, match_mask = mwpm_w_grad_v2(edges, weights, classes)
        
            if prediction == label:
                weights[match_mask] *= 0.8
                weights[~match_mask] *= 1.2
            else:
                weights[match_mask] *= 1.2
                weights[~match_mask] *= 0.8
                
            desired_weights[edge_map] = torch.tensor(weights)

        desired_weights = desired_weights.to(edge_weights.device)
        
        loss = loss_fun(edge_weights, desired_weights)
        ctx.save_for_backward(edge_weights, desired_weights)
        
        return loss

    @staticmethod
    def backward(
        ctx,
        grad_output,
    ):
        edge_weights, desired_edge_weights = ctx.saved_tensors
        grad = (edge_weights - desired_edge_weights) / edge_weights.shape[0]
        
        return None, grad, None, None, None


class SplitSyndromes(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, edges, edge_attr, detector_labels):

        node_range = torch.arange(0, detector_labels.shape[0]).to(edges.device)
        node_subset = node_range[detector_labels]

        valid_labels = torch.isin(edges, node_subset).sum(dim=0) == 2
        edges = edges[:, valid_labels]
        edge_attr = edge_attr[valid_labels, :]

        # now we want to remove the pairs (0-1, 1-0 etc)
        mask = edges[0, :] > edges[1, :]
        ind_range = torch.arange(edges.shape[1]).to(edges.device)
        edges, edge_attr = sort_edge_index(edges[:, ind_range[mask]], edge_attr[mask, :])
        
        return edges, edge_attr


class GraphNN(nn.Module):

    def __init__(
        self,
        hidden_channels_GCN=[32, 64, 128],
        hidden_channels_MLP=[128, 64],
        n_node_features=5,
    ):
        super().__init__()

        # GCN layers
        channels = [n_node_features] + hidden_channels_GCN
        self.graph_layers = nn.ModuleList(
            [
                nng.GraphConv(in_channels, out_channels)
                for (in_channels, out_channels) in zip(channels[:-1], channels[1:])
            ]
        )

        # Dense layers
        transition_dim = hidden_channels_GCN[-1] * 2 + 1
        channels = [transition_dim] + hidden_channels_MLP
        self.dense_layers = nn.ModuleList(
            [
                nn.Linear(in_channels, out_channels)
                for (in_channels, out_channels) in zip(channels[:-1], channels[1:])
            ]
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_channels_MLP[-1], 1)

        # Layer to split syndrome into X (Z)-graphs
        self.split_syndromes = SplitSyndromes()
        
        # Activation function
        self.activation = torch.nn.ReLU()

    def forward(
        self,
        x,
        edges,
        edge_attr,
        detector_labels,
        warmup=False,
    ):

        w = edge_attr[:, 0] * edge_attr[:, 1]
        for layer in self.graph_layers:
            x = layer(x, edges, w)
            x = self.activation(x)
        
        # split syndromes so only X (Z) nodes remain and create an edge embedding
        edges, edge_attr = self.split_syndromes(edges, edge_attr, detector_labels)
        x_src, x_dst = x[edges[0, :]], x[edges[1, :]]
        edge_feat = torch.cat([x_src, edge_attr[:, [0]], x_dst], dim=-1)
        
        # send the edge features through linear layers
        for layer in self.dense_layers:
            edge_feat = layer(edge_feat)
            edge_feat = self.activation(edge_feat)
        
        # output
        edge_feat = self.output_layer(edge_feat)
        
        # if warmup, train network to do identity mapping
        if warmup:
            label = edge_attr[:, [0]]
            return edge_feat, label
        
        # otherwise, save the edges with minimum weights (for each duplicate edge)
        n_edges = edge_feat.shape[0]
        edge_feat = torch.cat([edge_feat[::2], edge_feat[1::2]], dim=1)
        edge_classes = torch.stack([edge_attr[::2, 1], edge_attr[1::2, 1]], dim=1)
        
        min_inds = torch.argmin(edge_feat, dim=1)
        edge_feat = edge_feat[range(n_edges // 2), min_inds]
        edge_classes = edge_classes[range(n_edges // 2), min_inds]
        edges = edges[:, ::2]

        # normalise edge_weights
        edge_feat = torch.sigmoid(edge_feat)
        
        return edges, edge_feat, edge_classes