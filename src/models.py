from typing import Any
import torch
import torch.nn as nn
import torch_geometric.nn as nng
from torch_geometric.utils import sort_edge_index
from scipy.spatial.distance import cdist
import numpy as np
from qecsim.graphtools import mwpm
from src.graph import extract_edges
from signal import signal, SIGINT
import sys
# from src.utils import inference

def handler(signalnum, frame):
    sys.exit(0)


def mwpm_prediction(edges, weights, classes):

    signal(SIGINT, handler)
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
    if flip:
        gradient *= -1

    return flip, gradient

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
        factor: float = 1.5,
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

        return None, gradients, None, None, None, None, None

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
        grads = torch.zeros_like(edge_weights)

        for edges, weights, classes, edge_map in zip(edges_p_graph, weights_p_graph, classes_p_graph, edge_map_p_graph):
            edges = edges.cpu().numpy()
            weights = weights.cpu().numpy()
            classes = classes.cpu().numpy()

            prediction, gradients = mwpm_w_grad(edges, weights, classes)
            preds.append(prediction)
            grads[edge_map] = gradients.to(grads.device)


        preds = np.array(preds)

        # compute accuracy
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

        return None, grads, None, None, None, None, None


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
            x = torch.tanh(x)
        
        # split syndromes so only X (Z) nodes remain and create an edge embedding
        edges, edge_attr = self.split_syndromes(edges, edge_attr, detector_labels)
        x_src, x_dst = x[edges[0, :]], x[edges[1, :]]
        edge_feat = torch.cat([x_src, edge_attr[:, [0]], x_dst], dim=-1)
        
        # send the edge features through linear layers
        for layer in self.dense_layers:
            edge_feat = layer(edge_feat)
            edge_feat = torch.tanh(edge_feat)
        
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

        return edges, edge_feat, edge_classes

    

# class LocalSearch:
#     def __init__(self, model, search_radius, num_selections):
#         self.model = model
#         self.initial_score = torch.tensor(float(0))
#         self.top_score = self.initial_score
#         self.target = None
#         self.vector = torch.nn.utils.parameters_to_vector(model.parameters())
#         self.elite = self.vector.clone()
#         self.n = self.vector.numel()
#         self.running_idxs = np.arange(self.n)
#         np.random.shuffle(self.running_idxs)
#         self.idx = 0
#         self.value = []  # list of indices
#         self.num_selections = num_selections
#         self.magnitude = torch.empty(self.num_selections,
#                                 dtype=self.vector.dtype,
#                                 device=self.vector.device)
#         self.noise_vector = torch.empty_like(self.vector)
#         self.jumped = False
#         self.search_radius = search_radius

#     def set_value(self):
#         """Use the numpy choices function (which has no equivalent in Pytorch)
#         to generate a sample from the array of indices. The sample size and
#         distribution are dynamically updated by the algorithm's state.
#         """
#         self.check_idx()
#         choices = self.running_idxs[self.idx:self.idx+self.num_selections]
#         self.value = choices
#         self.idx+=self.num_selections

#     def check_idx(self):
#         if (self.idx+self.num_selections)>self.n:
#             self.idx=0
#             np.random.shuffle(self.running_idxs)

#     def set_noise(self):
#         # Cast to precision and CUDA, and edit shape
#         # search radius can be adjusted to fit scale of noise
#         self.magnitude.uniform_(-self.search_radius, self.search_radius).squeeze()

#     def set_noise_vector(self):
#         """ This function defines a noise tensor, and returns it. The noise
#         tensor needs to be the same shape as our originial vecotr. Hence, a
#         "basis" tensor is created with zeros, then the chosen indices are
#         modified.
#         """
#         self.noise_vector.fill_(0.)
#         self.noise_vector[self.value] = self.magnitude

#     def update_weights(self, model):
#         nn.utils.vector_to_parameters(self.vector, model.parameters())

#     def set_elite(self):
#         self.jumped = False
#         self.elite[self.value] = self.vector[self.value]
#             #self.elite.clamp_(-0.9, 0.9)
#             #self.elite.copy_(self.vector)
#         self.jumped = True
#             #self.frustration.reset_state()
        
#     def set_vector(self):
#         if not self.jumped:
#             #self.vector.copy_(self.elite)
#             elite_vals = self.elite[self.value]
#             self.vector[self.value] = elite_vals

#     def step(self,syndromes,flips):
#         #print(self.vector)
#         self.set_value()
#         self.set_noise()
#         self.set_noise_vector()
#         self.vector[torch.from_numpy(self.value)] = self.vector[torch.from_numpy(self.value)] + torch.from_numpy(self.value)
#         self.update_weights(self.model)
#         _, new_accuracy = inference(self.model,syndromes,flips)
#         if new_accuracy > self.top_score:
#             self.set_elite()
#             self.top_score = new_accuracy
#         else:
#             self.set_vector()
#         self.idx += 1
#             # decay to escape local maxima
#             #self.top_score -= 0.002


#     def return_topscore(self):
#         return self.top_score