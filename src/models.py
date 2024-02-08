from typing import Any
import torch
import torch.nn as nn
import torch_geometric.nn as nng
from scipy.spatial.distance import cdist
import numpy as np
import pymatching
from qecsim.graphtools import mwpm
from src.graph import extract_graphs

def mwpm_prediction(edges, weights, classes):

    # convert edges to dict
    classes = (classes > 0.0).astype(np.int32)
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
        return 1

class MWPMLoss(torch.autograd.Function):

    # experiment will be a 1-d array of same length as syndromes, indicating whether its a memory x or memory z-exp
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        edge_indx: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_labels: torch.Tensor,
        labels: np.ndarray,
        factor=1.2,
    ):

        # split edges and edge weights per syndrome
        (
            _,
            edges_p_graph,
            weights_p_graph,
            classes_p_graph,
            edge_map_p_graph,
        ) = extract_graphs(
            x,
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

            edges = edges.detach().numpy()
            weights = weights.detach().numpy()
            classes = classes.detach().numpy()

            prediction = mwpm_prediction(edges, weights, classes)
            preds.append(prediction)

            # we need a workaround for gradient computations
            preds_partial_de = []
            for i in range(edges.shape[1]):
                _weights = weights
                _weights[i] *= factor

                _classes = classes
                _classes[i] *= factor
                pred_w = mwpm_prediction(edges, _weights, classes)
                pred_c = mwpm_prediction(edges, weights, _classes)
                preds_partial_de.append([pred_w, pred_c])
                
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
        loss = 1 - accuracy

        ctx.save_for_backward(
            preds_grad,
            grad_help,
        )

        return torch.tensor(loss, requires_grad=True)

    @staticmethod
    def backward(
        ctx,
        grad_output,
    ):
        preds, grad_help = ctx.saved_tensors
        print(preds.shape, grad_help.shape)
        gradients = (grad_help[:, 0] - grad_help[:, 1])[:, None] * (preds - grad_help[:, 0][:, None])
        gradients.requires_grad = True
        return None, None, gradients, None, None, None

class EdgeNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.lin = nn.Linear(20, 20)
        
    def forward(self, x):
        x = self.lin(x)
        x_np = x.detach().numpy()
        
        with torch.no_grad():
            out = cdist(x_np, x_np)
        
        y = torch.tensor(out, requires_grad=True, dtype=torch.float32)
        return y
    
class MWPMLoss(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, syndrome, label, decoder: pymatching.Matching):
        
        # possible reshape and move to numpy
        # ...
        
        
        
        ctx.save_for_backward(syndrome, label)
        
        
        

class SimpleTest(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        pass
    

        
        
        

