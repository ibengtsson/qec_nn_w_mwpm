import torch
import numpy as np
import sys
sys.path.append("../")
from src.graph import extract_edges
from src.utils import mwpm_prediction, mwpm_w_grad, mwpm_w_grad_v2

from torch.multiprocessing import Pool
from torch.multiprocessing import cpu_count
from torch_geometric.utils import softmax

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
        eps = 1e-8
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
        
        preds = []
        for edges, weights, classes, edge_map, label in zip(edges_p_graph, weights_p_graph, classes_p_graph, edge_map_p_graph, labels):
            
            # # treat single edges seperately for now
            # if weights.shape[0] == 1:
            #     classes = classes.cpu().numpy().astype(np.int32)
            #     prediction = classes.sum() & 1
                
            #     if prediction == label: 
            #         desired_weights[edge_map] = 0
            #     else:
            #         desired_weights[edge_map] = 2
            #     preds.append(prediction)
            #     continue
            # take softmax of the inverse weight ---> gives probability that an edge should belong in matching
            #?????????????????????????
            
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
            preds.append(prediction)
        
        desired_weights = desired_weights.to(edge_weights.device)
        
        preds = np.array(preds)
        n_correct = (preds == labels).sum()
        accuracy = n_correct / labels.shape[0]
        
        loss = ((edge_weights - desired_weights) ** 2).mean()
        # loss = ((edge_weights - desired_weights) ** 2 * (1 - accuracy)).mean()
        # loss = -(desired_weights * torch.log(edge_weights) + (1 - desired_weights) * torch.log(1 - edge_weights)).mean()
        # loss = loss_fun(edge_weights, desired_weights)
        ctx.save_for_backward(edge_weights, desired_weights)
        ctx.accuracy = accuracy
        
        return loss

    @staticmethod
    def backward(
        ctx,
        grad_output,
    ):
        edge_weights, desired_edge_weights = ctx.saved_tensors
        accuracy = ctx.accuracy
        grad = edge_weights - desired_edge_weights
        # grad = grad * (1 - accuracy)
        
        return None, grad, None, None, None
    
class NestedMWPMLoss(torch.autograd.Function):

    # experiment will be a 1-d array of same length as syndromes, indicating whether its a memory x or memory z-exp
    @staticmethod
    def forward(
        ctx,
        *args,
    ):
        labels = args[-1]
        n_graphs = labels.shape[0]
        
        edge_index = args[:n_graphs]
        edge_weights = args[n_graphs:(2*n_graphs)]
        edge_classes = args[(2*n_graphs):-1]
        
        # we must loop through every graph since each one will have given a new set of edge weights
        loss = 0
        n_edges = 0
        preds = []
        grads = []
        for edges, weights, classes, label in zip(edge_index, edge_weights, edge_classes, labels):

            # find which (of the two) egdes that have the minimum weight for each node pair
            n_edges += edges.shape[1]
            weights = torch.cat([weights[::2], weights[1::2]], dim=1)
            classes = torch.cat([classes[::2], classes[1::2]], dim=1)
            edges = edges[:, ::2]
  
            # initialise a gradient array with same block shape
            grad = torch.zeros_like(weights)
            
            weights, min_inds = torch.min(weights, dim=1)
            classes = classes[range(edges.shape[1]), min_inds]
            
            # do a softmax on the weights
            weights = torch.nn.functional.sigmoid(weights)

            # move to CPU and run MWPM
            edges = edges.cpu().numpy()
            _weights = weights.cpu().numpy()
            classes = classes.cpu().numpy()
            
            prediction, match_mask = mwpm_w_grad_v2(edges, _weights, classes)
            
            # compute gradients
            optimal_edge_weights = torch.zeros_like(weights)
            if prediction == label:
                optimal_edge_weights[~match_mask] = 1
                optimal_edge_weights[match_mask] = 0
            else:
                optimal_edge_weights[match_mask] = 1
                optimal_edge_weights[~match_mask] = 0
                
            diff = weights - optimal_edge_weights
            grad[range(edges.shape[1]), min_inds] = diff
            grads.append(grad.T.flatten()[:, None])
            
            # compute loss and save prediction
            first_log = torch.clamp(torch.log(weights), min=-100, max=None)
            second_log = torch.clamp(torch.log(1 - weights), min=-100, max=None)
            loss += (-(optimal_edge_weights * first_log + (1 - optimal_edge_weights) * second_log)).sum()
            preds.append(prediction)
            
        grads = torch.nested.nested_tensor(grads)
        preds = np.array(preds)
        
        # compute accuracy and scale loss
        n_correct = (preds == labels).sum()
        accuracy = n_correct / labels.shape[0]

        loss = loss / n_edges
        ctx.save_for_backward(grads)
        ctx.accuracy = accuracy
        ctx.n_graphs = labels.shape[0]

        return loss

    @staticmethod
    def backward(
        ctx,
        grad_output,
    ):
        grads, = ctx.saved_tensors
        accuracy = ctx.accuracy
        n_graphs = ctx.n_graphs
        # grads = grads * np.sqrt(1 - accuracy)

        return (None,) * n_graphs + grads.unbind() + (None,) * n_graphs + (None,)
    
class MWPMLoss_v4(torch.autograd.Function):

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
        edge_weights = torch.nn.functional.sigmoid(edge_weights)
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
        
        # weight for matching inbalance
        bias_reversal = torch.zeros_like(edge_weights, device="cpu")
        
        # class weighting
        identity_w = (1 / (labels == 0).sum()) * (labels.shape[0] / 2.0)
        flip_w = (1 / (labels == 1).sum()) * (labels.shape[0] / 2.0)
        class_weight = {0: identity_w, 1: flip_w}

        preds = []
        wrong_inds = []
        for i, (edges, weights, classes, edge_map, label) in enumerate(zip(edges_p_graph, weights_p_graph, classes_p_graph, edge_map_p_graph, labels)):
            
            edges = edges.cpu().numpy()
            weights = weights.cpu().numpy()
            classes = classes.cpu().numpy()
            edge_map = edge_map.cpu()

            prediction, match_mask = mwpm_w_grad_v2(edges, weights, classes)
            _desired_weights = torch.zeros(weights.shape)
            _bias_reversal = torch.zeros(weights.shape)
            
            if prediction == label:
                _desired_weights[~match_mask] = 1
                _desired_weights[match_mask] = 0

            else:
                
                # find a label        
                if edges.shape[1] > 1:
                    n = 30
                    k = 0
                    while (prediction != label and k <  n):
                        w = np.random.rand(weights.shape[0])
                        prediction, trial_mask = mwpm_w_grad_v2(edges, w, classes)
                        k += 1
                    if k < n:
                        _desired_weights[~trial_mask] = 1
                        _desired_weights[trial_mask] = 0
                    else:
                        # default
                        _desired_weights[match_mask] = 1
                        _desired_weights[~match_mask] = 0
                else:
                    # default
                    _desired_weights[match_mask] = 1
            
            _bias_reversal[~match_mask] = edges.shape[1] / np.maximum((~match_mask).sum(), 1)
            _bias_reversal[match_mask] = edges.shape[1] / np.maximum(match_mask.sum(), 1)
            desired_weights[edge_map] = _desired_weights

            bias_reversal[edge_map] = _bias_reversal * class_weight[label]
            preds.append(prediction)
            
           

        desired_weights = desired_weights.to(edge_weights.device)
        bias_reversal = bias_reversal.to(edge_weights.device)
        

        first_log = torch.clamp(torch.log(edge_weights), min=-100, max=None)
        second_log = torch.clamp(torch.log(1 - edge_weights), min=-100, max=None)
        loss = (-(desired_weights * first_log + (1 - desired_weights) * second_log) * bias_reversal).mean() 

        ctx.save_for_backward(edge_weights, desired_weights, bias_reversal)
        
        return loss, wrong_inds

    @staticmethod
    def backward(
        ctx,
        grad_output,
        _0,
    ):
        edge_weights, desired_edge_weights, bias_reversal = ctx.saved_tensors
        grad = (edge_weights - desired_edge_weights) * bias_reversal
        return None, grad, None, None, None

def loss_help_wrapper(args):
    print("Thread start")
    return loss_help(*args)

def loss_help(edges, weights, classes, edge_map, label):
    
    edges = edges.numpy()
    _weights = weights.numpy()
    classes = classes.numpy()

    prediction, match_mask = mwpm_w_grad_v2(edges, _weights, classes)

    desired_weights = torch.zeros_like(weights)
    if prediction == label:
        desired_weights[~match_mask] = 1
        desired_weights[match_mask] = 0
    else:
        desired_weights[match_mask] = 1
        desired_weights[~match_mask] = 0
    
    loss = (-(desired_weights * torch.log(weights) + (1 - desired_weights) * torch.log(1 - weights))).sum()
    grad = weights - desired_weights

    return loss, grad, edge_map
    
class MWPMLoss_v4_parallel(torch.autograd.Function):

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
        edge_weights = torch.nn.functional.sigmoid(edge_weights)
        edge_attr = torch.stack([edge_weights, edge_classes], dim=1)
        
        # move data from CUDA
        edge_indx = edge_indx.cpu()
        edge_attr = edge_attr.cpu()
        batch_labels = batch_labels.cpu()
        
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
        
        # run mwpm in parallel
        loss = 0
        grads = torch.zeros(edge_weights.shape)
        n_processes = cpu_count() - 1
        pool = Pool(processes=n_processes)
        chunksize = int(0.8 * len(edges_p_graph) / n_processes)
        for res in pool.imap_unordered(loss_help_wrapper, list(zip(edges_p_graph, weights_p_graph, classes_p_graph, edge_map_p_graph, labels)), chunksize=chunksize):
            _loss, grad, grad_map = res
            grads[grad_map] = grad
            loss += _loss

        loss /= edge_weights.shape[0]
        loss = loss.to(edge_weights.device)
        grads = grads.to(edge_weights.device)
        ctx.save_for_backward(grads)
        
        return loss

    @staticmethod
    def backward(
        ctx,
        grad_output,
    ):
        grad, = ctx.saved_tensors
        
        return None, grad, None, None, None

class AttentionMWPMLoss(torch.autograd.Function):

    # experiment will be a 1-d array of same length as syndromes, indicating whether its a memory x or memory z-exp
    @staticmethod
    def forward(
        ctx,
        edge_indx: torch.Tensor,
        edge_weights: torch.Tensor,
        edge_classes: torch.Tensor,
        labels: np.ndarray,
    ):

        # send weights through a sigmoid
        edge_weights = torch.nn.functional.sigmoid(edge_weights) 
        
        # initialise gradients
        grads = torch.zeros_like(edge_weights, device="cpu")
        
        # class weighting
        identity_w = (1 / np.maximum((labels == 0).sum(), 1)) * (labels.shape[0] / 2.0)
        flip_w = (1 / np.maximum((labels == 1).sum(), 1)) * (labels.shape[0] / 2.0)
        class_weight = {0: identity_w, 1: flip_w}

        loss = 0
        wrong_inds = []
        test = []
        for i, (edges, weights, classes, label) in enumerate(zip(edge_indx, edge_weights, edge_classes, labels)):
            
            # begin by removing the trailing zeros from the padding
            mask = edges[:, 0] != edges[:, 1]
            edges = edges[mask, :].T
            weights = weights[mask].cpu()
            classes = classes[mask]

            edges = edges.cpu().numpy()
            _weights = weights.numpy().squeeze()
            classes = classes.cpu().numpy().squeeze()

            prediction, match_mask = mwpm_w_grad_v2(edges, _weights, classes)
            desired_weights = torch.zeros(weights.shape)
            matching_compensation = torch.zeros(weights.shape)
            
            if prediction == label:
                desired_weights[~match_mask] = 1
                desired_weights[match_mask] = 0
                
            else:
            
                # find a label        
                if edges.shape[1] > 1:
                    n = 30
                    k = 0
                    while (prediction != label and k <  n):
                        w = np.random.rand(weights.shape[0])
                        prediction, trial_mask = mwpm_w_grad_v2(edges, w, classes)
                        k += 1
                    if k < n:
                        desired_weights[~trial_mask] = 1
                        desired_weights[trial_mask] = 0
                    else:
                        # default
                        desired_weights[match_mask] = 1
                        desired_weights[~match_mask] = 0
                else:
                    # default
                    desired_weights[match_mask] = 1
                
                # save indices to wrong predictions
                wrong_inds.append(i)
            
            
            matching_compensation[~match_mask] = edges.shape[1] / np.maximum((~match_mask).sum(), 1)
            matching_compensation[match_mask] = edges.shape[1] / np.maximum(match_mask.sum(), 1)
            
            # loss
            first_log = torch.clamp(torch.log(weights), min=-100, max=None)
            second_log = torch.clamp(torch.log(1 - weights), min=-100, max=None)
            loss += (-(desired_weights * first_log + (1 - desired_weights) * second_log) * matching_compensation * class_weight[label]).sum()
            
            # gradients
            grads[i, mask] = (weights - desired_weights) * matching_compensation * class_weight[label]
   
        # move gradients and loss to GPU
        grads = grads.to(edge_weights.device)
        loss = loss.to(edge_weights.device)
        
        loss /= edge_indx.shape[0]
        
        ctx.save_for_backward(grads)
        
        return loss, wrong_inds

    @staticmethod
    def backward(
        ctx,
        grad_output,
        _0,
    ):
        grads, = ctx.saved_tensors

        return None, grads, None, None
    
class AttentionMWPMLossV2(torch.autograd.Function):

    # experiment will be a 1-d array of same length as syndromes, indicating whether its a memory x or memory z-exp
    @staticmethod
    def forward(
        ctx,
        edge_indx: torch.Tensor,
        edge_weights: torch.Tensor,
        edge_classes: torch.Tensor,
        labels: np.ndarray,
    ):
        
        # initialise gradients
        grads = torch.zeros_like(edge_weights, device="cpu")
        
        # class weighting
        identity_w = (1 / (labels == 0).sum()) * (labels.shape[0] / 2.0)
        flip_w = (1 / (labels == 1).sum()) * (labels.shape[0] / 2.0)
        class_weight = {0: identity_w, 1: flip_w}

        loss = 0
        wrong_inds = []
        edge_limits = [0, 1e3]
        for i, (edges, weights, classes, label) in enumerate(zip(edge_indx, edge_weights, edge_classes, labels)):
            
            # begin by removing the zeros from the padding 
            mask = edges[:, 0] != edges[:, 1]
            edges = edges[mask, :].T

            weights = weights[mask]
            classes = classes[mask]

            edges = edges.cpu().numpy()
            _weights = weights.cpu().numpy().squeeze()
            classes = classes.cpu().numpy().squeeze()
            
    
            prediction, match_mask = mwpm_w_grad_v2(edges, _weights, classes)
            desired_weights = torch.zeros(weights.shape)
            matching_compensation = torch.zeros(weights.shape)
            
            if prediction == label:
                desired_weights[~match_mask] = edge_limits[1]
                desired_weights[match_mask] = edge_limits[0]
                
            else:
                desired_weights[match_mask] = edge_limits[1]
                desired_weights[~match_mask] = edge_limits[0]
                
                # save indices to wrong predictions
                wrong_inds.append(i)
            
            
            matching_compensation[~match_mask] = edges.shape[1] / np.maximum((~match_mask).sum(), 1)
            matching_compensation[match_mask] = edges.shape[1] / np.maximum(match_mask.sum(), 1)
            
            # loss
            loss += ((desired_weights - weights) ** 2 / weights.shape[0]).sum()
            
            # gradients
            grads[i, mask] = (weights - desired_weights) * matching_compensation * class_weight[label]
   
        # move gradients and loss to GPU
        grads = grads.to(edge_weights.device)
        loss = loss.to(edge_weights.device)

        loss /= edge_indx.shape[0] * edge_indx.shape[1]
        
        ctx.save_for_backward(grads)
        
        return loss, wrong_inds

    @staticmethod
    def backward(
        ctx,
        grad_output,
        _0,
    ):
        grads, = ctx.saved_tensors
        return None, grads, None, None
