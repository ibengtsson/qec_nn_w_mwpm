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
        preds_grad = torch.zeros_like(edge_attr)
        grad_help = torch.zeros_like(edge_attr)

        for i, (edges, weights, classes, edge_map) in enumerate(
            zip(edges_p_graph, weights_p_graph, classes_p_graph, edge_map_p_graph)
        ):
            # print(edges)
            # edges = edges.detach().numpy()
            edges = edges.numpy()
            # weights = weights.detach().numpy()
            weights = weights.numpy()
            # classes = classes.detach().numpy()
            classes = classes.numpy()

            prediction = mwpm_prediction(edges, weights, classes)
            # print(f"{prediction=}")
            preds.append(prediction)

            # we need a workaround for gradient computations
            preds_partial_de = []
            for j in range(edges.shape[1]):
                _weights = weights.copy()
                # _weights[j] *= factor
                # _weights[j] += factor
                # _weights[j] = _weights[j] + delta
                _weights[j] = _weights[j] * factor
                delta = _weights[j] - weights[j]
   
                # _classes = classes
                # _classes[j] *= factor
                # _classes[j] += factor
                # _classes[j] = _classes[j] + delta
                pred_w = mwpm_prediction(edges, _weights, classes)
                # print(f"{pred_w=}")
                # pred_c = mwpm_prediction(edges, weights, _classes)
                # preds_partial_de.append(pred_w)
                preds_partial_de.append([pred_w, delta])
                
            # REMOVE WHEN WE KNOW THAT ALL SYNDROMES HAVE AN EDGE
            if edge_map.numel() == 0:
                continue
            else:
                preds_grad[edge_map, :] = torch.tensor(
                    preds_partial_de, dtype=torch.float32
                )
                grad_help[edge_map, 0] = prediction
                grad_help[edge_map, 1] = labels[i]
        preds = np.array(preds)

        # compute accuracy
        n_correct = (preds == labels).sum()
        accuracy = n_correct / labels.shape[0]
        loss = torch.tensor(1 - accuracy, requires_grad=True)
        # loss.retain_grad()

        ctx.save_for_backward(
            preds_grad,
            grad_help,
        )
        # ctx.delta = delta
        
        return loss

    @staticmethod
    def backward(
        ctx,
        grad_output,
    ):
        shift_preds, grad_help = ctx.saved_tensors
        delta = shift_preds[:, 1]
        shift_preds = shift_preds[:, 0]
        
        preds = grad_help[:, 0]
        labels = grad_help[:, 1]
        # delta = ctx.delta
        # gradients = (preds - labels) * (shift_preds - preds) / delta
        gradients = (0.5 * (shift_preds + preds) - labels) * (shift_preds - preds) / delta
        # gradients = 0.5 * ((preds - grad_help[:, 1][:, None]) - torch.abs((preds - grad_help[:, 1][:, None] - 1)))
        # gradients = (preds - grad_help[:, 0][:, None] - 0.5) / delta
        # gradients = torch.ones_like(preds)
        # gradients = torch.randn_like(preds)
        # gradients = torch.abs((preds - labels) + (shift_preds - labels) - 1)
        
        # gradients[gradients > 0] = 1
        # gradients[gradients < 1] = -0.5
        # gradients[gradients < 0] = -1
        # gradients[gradients == 0] = 1
        print(f"{torch.count_nonzero(gradients)=}")
        gradients.requires_grad = True
        # print(gradients[:10, :])
        # print(gradients[:100])
        
        return None, gradients, None, None, None, None, None
    
class SplitSyndromes(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, edges, edge_attr, detector_labels):

        node_range = torch.arange(0, detector_labels.shape[0])
        node_subset = node_range[detector_labels]
        
        valid_labels = torch.isin(edges, node_subset).sum(dim=0) == 2
        edges = edges[:, valid_labels]
        edge_attr = edge_attr[valid_labels, :]
        
        return edges, edge_attr

    
    

        
        
        

