import torch
import torch.nn as nn
import torch_geometric.nn as nng
from scipy.spatial.distance import cdist
import stim
import numpy as np
import pymatching as pm
import matplotlib.pyplot as plt
from qecsim.graphtools import mwpm
import operator
from torch_geometric.utils import coalesce, is_undirected, sort_edge_index

import sys

sys.path.append("../")
from src.simulations import SurfaceCodeSim
from src.graph import get_batch_of_graphs, extract_edges, extract_edges_v2
from src.utils import time_it


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


class MWPM:

    def __init__(self, circuit: stim.Circuit, detector_dict: dict):
        self.decoder = pm.Matching.from_stim_circuit(circuit)
        self.detector_dict = detector_dict

    def update_edge(self, edge, weight):
        self.decoder.add_edge(edge[0], edge[1], weight=weight, merge_strategy="replace")

    def update_edges(self, nodes, edges, edge_weights):

        # only care about edges belonging to exp
        for nodes, weight in zip(edges.T, edge_weights):
            self.decoder.add_edge(
                nodes[0], nodes[1], weight=weight, merge_strategy="replace"
            )

    def decode_batch(self, syndromes):
        return self.decoder.decode_batch(syndromes)

    def decode(self, syndrome):
        return self.decoder.decode(syndrome)

    # should replace edge weights to their initial value
    def reset(self):
        pass


# naive implementation
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
        gradients = (grad_help[:, 0] - grad_help[:, 1])[:, None] * (preds - grad_help[:, 0][:, None])
        gradients.requires_grad = True
        return None, None, gradients, None, None, None


class GATGNN(nn.Module):

    def __init__(self, n_heads=1, edge_dimensions=1):
        super().__init__()

        self.gat1 = nng.GATv2Conv(
            -1,
            16,
            heads=n_heads,
            concat=False,
            edge_dim=edge_dimensions,
            add_self_loops=False,
        )
        self.gat2 = nng.GATv2Conv(
            16,
            32,
            heads=n_heads,
            concat=False,
            edge_dim=edge_dimensions,
            add_self_loops=False,
        )

    def forward(self, x, edges, edge_weights):

        x, (_, edge_weights) = self.gat1(
            x, edges, edge_weights, return_attention_weights=True
        )
        x = torch.nn.functional.relu(x, inplace=True)
        x, (edges, edge_weights) = self.gat2(
            x, edges, edge_weights, return_attention_weights=True
        )

        return edges, edge_weights


class TransformerGNN(nn.Module):

    def __init__(self, n_heads=1, edge_dimensions=1):
        super().__init__()

        self.t1 = nng.TransformerConv(
            -1,
            16,
            heads=n_heads,
            concat=False,
            edge_dim=edge_dimensions,
            add_self_loops=False,
        )
        self.t2 = nng.TransformerConv(
            16,
            32,
            heads=n_heads,
            concat=False,
            edge_dim=edge_dimensions,
            add_self_loops=False,
        )

    def forward(self, x, edges, edge_weights):
        x, (_, edge_weights) = self.t1(
            x, edges, edge_weights, return_attention_weights=True
        )
        x = torch.nn.functional.relu(x, inplace=True)
        x, (edges, edge_weights) = self.t2(
            x, edges, edge_weights, return_attention_weights=True
        )

        return edges, edge_weights

def extract_graphs(x, edges, edge_attr, batch_labels):

    node_range = torch.arange(0, x.shape[0])

    nodes_per_syndrome = []
    edges_per_syndrome = []
    weights_per_syndrome = []
    classes_per_syndrome = []
    edge_indx = []
    edge_weights = edge_attr[:, 0]
    edge_classes = edge_attr[:, 1]
    for i in range(batch_labels[-1] + 1):
        ind_range = torch.nonzero(batch_labels == i)

        # nodes
        nodes_per_syndrome.append(x[ind_range])

        # edges
        edge_mask = (edges >= node_range[ind_range[0]]) & (
            edges <= node_range[ind_range[-1]]
        )
        new_edges = edges[:, edge_mask[0, :]] - node_range[ind_range[0]]
        new_weights = edge_weights[edge_mask[0, :]]
        new_edge_classes = edge_classes[edge_mask[0, :]]

        edges_per_syndrome.append(new_edges)
        weights_per_syndrome.append(new_weights)
        classes_per_syndrome.append(new_edge_classes)

        edge_range = torch.arange(0, edges.shape[1])
        edge_indx.append(edge_range[edge_mask[0, :]])

    return (
        nodes_per_syndrome,
        edges_per_syndrome,
        weights_per_syndrome,
        classes_per_syndrome,
        edge_indx,
    )

class SplitSyndromes(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, edges, edge_attr, detector_labels):

        node_range = torch.arange(0, detector_labels.shape[0])
        node_subset = node_range[detector_labels]
        
        valid_labels = torch.isin(edges, node_subset).sum(dim=0) == 2
        return edges[:, valid_labels], edge_attr[valid_labels, :]

def main():

    reps = 5
    code_sz = 5
    p = 1e-3
    n_shots = 10000

    sim = SurfaceCodeSim(reps, code_sz, p, n_shots)
    syndromes, flips, _ = sim.generate_syndromes(use_for_mwpm=True)
    x, edges, edge_attr, batch_labels, detector_labels = get_batch_of_graphs(syndromes, 20)
    time_it(extract_edges, 2, edges, edge_attr, batch_labels)
    time_it(extract_edges_v2, 2, edges, edge_attr, batch_labels)

    
    
    return 

def test_nn():

    n_nodes = 64
    node_dimensions = 4

    edge_dimensions = 1

    x = torch.randn((n_nodes, node_dimensions))
    node_range = torch.arange(0, n_nodes)
    edges = torch.tensor([[0, 1, 8, 1, 0, 0, 45, 22], [1, 3, 18, 7, 7, 6, 5, 2]])
    weights = torch.randn((edges.shape[1], 1))

    gat_model = GATGNN()
    edges, weights_new = gat_model(x, edges, weights)

    print(f"{weights[:10]=}")
    print(f"{weights_new[:10]=}")

    trans_model = TransformerGNN()
    edges, weights_new = trans_model(x, edges, weights)

    print(f"{weights[:10]=}")
    print(f"{weights_new[:10]=}")


def py_match():

    H = np.array(
        [
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        ]
    )

    matching = pm.Matching.from_check_matrix(H, repetitions=1)
    print(matching)
    matching.draw()
    plt.show()


def stim_mwpm():

    reps = 1
    code_sz = 5
    p = 1e-1
    n_shots = 100
    # sim = SurfaceCodeSim(reps, code_sz, p, n_shots)
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=reps,
        distance=code_sz,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )

    sampler = circuit.compile_detector_sampler()
    detectors, flips = sampler.sample(n_shots, separate_observables=True)

    det_coords = circuit.get_detector_coordinates()
    det_coords = np.array(list(det_coords.values()))

    # rescale space like coordinates:
    det_coords[:, :2] = det_coords[:, :2] / 2

    # convert to integers
    det_coords = det_coords.astype(np.uint8)

    xz_map = (np.indices((code_sz + 1, code_sz + 1)).sum(axis=0) % 2).astype(bool)
    det_indx = np.arange(det_coords.shape[0])
    x_or_z = np.array([xz_map[cord[0], cord[1]] for cord in det_coords])

    x_dict = dict(
        [
            (tuple(cord), ind)
            for cord, ind in zip(det_coords[x_or_z, :], det_indx[x_or_z])
        ]
    )
    z_dict = dict(
        [
            (tuple(cord), ind)
            for cord, ind in zip(det_coords[~x_or_z, :], det_indx[~x_or_z])
        ]
    )

    detector_dict = {}
    detector_dict["x"] = x_dict
    detector_dict["z"] = z_dict

    sz = code_sz + 1

    syndrome_x = np.zeros((sz, sz), dtype=np.uint8)
    syndrome_x[::2, 1 : sz - 1 : 2] = 1
    syndrome_x[1::2, 2::2] = 1

    syndrome_z = np.rot90(syndrome_x) * 3

    mask = np.dstack([syndrome_x + syndrome_z] * (reps + 1))
    mask = np.repeat(mask[None, ...], detectors.shape[0], 0)
    syndromes = np.zeros_like(mask)
    syndromes[:, det_coords[:, 1], det_coords[:, 0], det_coords[:, 2]] = detectors

    # syndromes[..., 1:] = (syndromes[..., 1:] - syndromes[..., 0:-1]) % 2
    syndromes[np.nonzero(syndromes)] = mask[np.nonzero(syndromes)]

    sim = SurfaceCodeSim(
        reps, code_sz, p, n_shots, code_task="surface_code:rotated_memory_z"
    )
    detectors, syndromes, flips, _ = sim.generate_syndromes(n_shots)

    for i in range(20):
        if flips[i] == 1:
            print(syndromes[i, :, :, 0])
            print(detectors[i, :12])
            print(syndromes[i, :, :, 1])
            print(detectors[i, 12:])
            print(syndromes[i, :, :, :].sum(axis=-1))

    print(detector_dict["z"])
    return

    matching = pm.Matching.from_stim_circuit(circuit)
    # print(syndromes[0].shape)
    preds = matching.decode(syndromes[0])
    # print(preds)

    # print(matching)
    # print(matching.edges())

    # print(circuit)

    det_coords = circuit.get_detector_coordinates()
    det_coords = np.array(list(det_coords.values()))

    # rescale space like coordinates:
    det_coords[:, :2] = det_coords[:, :2] / 2
    # convert to integers
    det_coords = det_coords.astype(np.uint8)

    xz_map = (np.indices((code_sz + 1, code_sz + 1)).sum(axis=0) % 2).astype(bool)
    det_indx = np.arange(det_coords.shape[0])
    x_or_z = np.array([xz_map[cord[0], cord[1]] for cord in det_coords])

    x_dict = dict(
        [
            (tuple(cord), ind)
            for cord, ind in zip(det_coords[x_or_z, :], det_indx[x_or_z])
        ]
    )
    z_dict = dict(
        [
            (tuple(cord), ind)
            for cord, ind in zip(det_coords[~x_or_z, :], det_indx[~x_or_z])
        ]
    )

    detectors = {}
    detectors["x"] = x_dict
    detectors["z"] = z_dict

    print(detectors)
    # import matplotlib.pyplot as plt

    # matching.draw()
    # plt.show()


def graphs():

    reps = 1
    code_sz = 3
    p = 1e-1
    n_shots = 100
    sim = SurfaceCodeSim(
        reps, code_sz, p, n_shots, code_task="surface_code:rotated_memory_z"
    )
    syndromes, flips, _ = sim.sample_syndromes(n_shots)

    det_cords, _ = sim.get_detector_coords()
    mask = sim.syndrome_mask()
    mask = np.repeat(mask[None, ...], syndromes.shape[0], 0)
    print(mask.shape)

    test = np.zeros_like(mask)
    test[:, det_cords[:, 1], det_cords[:, 0], det_cords[:, 2]] = syndromes
    print(test[0, :, :, 0])
    print(test[0, :, :, 1])

    test[..., 1:] = (test[..., 1:] - test[..., 0:-1]) % 2
    print(test[0, :, :, 0])
    print(test[0, :, :, 1])
    return
    for i in range(50):
        if flips[i] == 1:
            print(syndromes[i, :, :, :].sum(-1))
        # print(flips[i])
    print(det_cords)

    x, edge_index, edge_attr, batch_labels = get_batch_of_graphs(
        syndromes, m_nearest_nodes=5
    )

    nodes_per_syndrome, edges_per_syndrome, weights_per_syndrome = extract_graphs(
        x,
        edge_index,
        edge_attr,
        batch_labels,
    )


def check_simulations():
    reps = 7
    code_sz = 11
    p = 5e-3
    n_shots = 50000
    sim = SurfaceCodeSim(
        reps, code_sz, p, n_shots, code_task="surface_code:rotated_memory_z"
    )
    dets, syndromes, flips, _ = sim.generate_syndromes(n_shots)
    n_nodes = np.count_nonzero(syndromes)
    n_nodes = np.count_nonzero(dets)
    print(f"We have on average {n_nodes / n_shots:.2f} nodes.")


if __name__ == "__main__":
    # stim_mwpm()
    main()
    # test_nn()
    # graphs()
    # py_match()
    # check_simulations()
