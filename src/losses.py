import torch
import numpy as np
import sys
sys.path.append("../")
from src.graph import extract_edges
from src.utils import mwpm_prediction, mwpm_w_grad, mwpm_w_grad_v2

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

        # initialise
        loss_fun = torch.nn.MSELoss()
        edge_attr = torch.stack([edge_weights, edge_classes], dim=1)
        
        # class imbalance
        n_no_flips = labels.shape[0] - labels.sum()
        n_flips = labels.sum()
        class_weight = [1 / (n_no_flips / labels.shape[0]), 1 / (n_flips / labels.shape[0])]
        
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
        class_balancer = torch.ones_like(edge_weights, device="cpu")
        
        preds = []
        for edges, weights, classes, edge_map, label in zip(edges_p_graph, weights_p_graph, classes_p_graph, edge_map_p_graph, labels):
            edges = edges.cpu().numpy()
            weights = weights.cpu().numpy()
            classes = classes.cpu().numpy()
            edge_map = edge_map.cpu()

            prediction, match_mask = mwpm_w_grad_v2(edges, weights, classes)

            _desired_weights = torch.zeros(weights.shape)
            if prediction == label:
                norm = max((~match_mask).sum(), 1)
                _desired_weights[~match_mask] = 1 / norm
                _desired_weights[match_mask] = 0
            else:
                norm = max((match_mask).sum(), 1)
                _desired_weights[match_mask] = 1 / norm
                _desired_weights[~match_mask] = 0
                
            desired_weights[edge_map] = _desired_weights
            class_balancer[edge_map] *= class_weight[label]
            preds.append(prediction)

        desired_weights = desired_weights.to(edge_weights.device)
        class_balancer = class_balancer.to(edge_weights.device)
        
        preds = np.array(preds)
        n_correct = (preds == labels).sum()
        accuracy = n_correct / labels.shape[0]
        
        # loss = ((edge_weights - desired_weights) ** 2 * class_balancer).mean()
        loss = ((edge_weights - desired_weights) ** 2 * (1 - accuracy)).mean()
        
        # loss = loss_fun(edge_weights, desired_weights)
        ctx.save_for_backward(edge_weights, desired_weights, class_balancer)
        ctx.accuracy = accuracy
        
        return loss

    @staticmethod
    def backward(
        ctx,
        grad_output,
    ):
        edge_weights, desired_edge_weights, class_balancer = ctx.saved_tensors
        accuracy = ctx.accuracy
        grad = (edge_weights - desired_edge_weights) / edge_weights.shape[0]
        grad = grad * (1 - accuracy)
        grad.requires_grad = True
        
        # grad = torch.ones_like(edge_weights, requires_grad=True)
        
        return None, grad, None, None, None