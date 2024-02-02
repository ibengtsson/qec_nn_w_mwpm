import torch
import torch.nn as nn
import torch_geometric.nn as nng
from scipy.spatial.distance import cdist
import stim
import numpy as np
import pymatching as pm

import sys
sys.path.append("../")
from src.simulations import SurfaceCodeSim

class MWPM():
    
    def __init__(self, circuit: stim.Circuit=None):
        self.decoder = pm.Matching.from_stim_circuit(circuit)

    def update_edge(self, edge, weight):
        self.decoder.add_edge(edge[0], edge[1], weight=weight, merge_strategy="replace")
        
    def update_edges(self, edges, edge_weights):
        for nodes, weight in zip(edges.T, edge_weights):
            self.decoder.add_edge(nodes[0], nodes[1], weight=weight, merge_strategy="replace")
        
    def decode_batch(self, syndromes):
        return self.decoder.decode_batch(syndromes)

    def decode(self, syndrome):
        return self.decoder.decode(syndrome)
    
    def reset(self):
        pass

# naive implementation
class MWPMLoss(torch.autograd.Function):
    
    @staticmethod
    def forward(
        ctx, 
        edge_weights: torch.Tensor,
        syndromes, 
        edges: torch.Tensor, 
        labels: np.ndarray, 
        circuit: stim.Circuit,
        delta=1,
        ):
        
        # create decoder
        decoder = MWPM(circuit)
        edges = edges.detach().numpy()
        edge_weights = edge_weights.detach().numpy()
        
        # we must loop through every graph since each one will have given a new set of edge weights
        # MUST FIX!
        preds = []
        preds_grad = []
        for syndrome in syndromes:
            
            # until fixed, add noise to edge weights
            noise = np.random.randn(*edge_weights.shape)
            decoder.update_edges(edges, edge_weights + noise)
            prediction = decoder.decode(syndrome)
            
            preds.append(prediction)
            
            # we need a workaround for gradient computations
            shifted_edge_weights = edge_weights + delta
            
            preds_partial_de = []
            for edge, weight in zip(edges.T, shifted_edge_weights):
                decoder.update_edge(edge, weight)
                prediction = decoder.decode(syndrome)
                
                preds_partial_de.append(prediction)
            preds_grad.append(np.array(preds_partial_de))
            
        preds = np.array(preds)
        preds_grad = np.array(preds_grad)
        
        # compute accuracy      
        n_correct = np.sum(np.any(preds == labels, axis=1))
        accuracy = n_correct / labels.shape[0]
        loss = 1 - accuracy
        
        ctx.save_for_backward(torch.tensor(preds, dtype=torch.float32), torch.tensor(preds_grad, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32), torch.tensor(delta))
        
        return torch.tensor(loss, requires_grad=True)
    
    @staticmethod
    def backward(
        ctx,
        grad_output,
    ):
        preds, preds_grad, labels, delta = ctx.saved_tensors
        gradients = torch.zeros(preds_grad.shape, requires_grad=True)
        
        # NEEDS A FIX TO ENSURE GRADIENTS ARE ADDED CORRECTLY! NOW ITS ASSUMED ITS ALWAYS THE SAME EDGE WEIGHTS THAT ARE PASSED THROUGH NN
        for prediction, label, prediction_de in zip(preds, labels, preds_grad):
            grad = (prediction - label) * (prediction_de - prediction) / delta
            
            # THIS PART MUST BE FIXED
            gradients += grad
        gradients /= prediction.shape[0]
        return None, None, gradients, None, None, None

def main():
    
    syndromes = [True, False, False, True, False, False, False, True]
    edges = torch.randint(0, 8, size=(2, 10), dtype=torch.int32)
    weights = torch.randn(size=(1, 20), requires_grad=True).T
    label = np.ones(shape=(20, 1))
    
    reps = 1
    code_sz = 3
    p = 1e-1
    n_shots = 10000
    circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=reps,
            distance=code_sz,
            after_clifford_depolarization=0,
            after_reset_flip_probability=0,
            before_measure_flip_probability=p,
            before_round_data_depolarization=0,
        )
    sampler = circuit.compile_detector_sampler()
    syndromes, flips = sampler.sample(n_shots, separate_observables=True)
    loss_fun = MWPMLoss.apply
    loss = loss_fun(weights, syndromes, edges, np.array(flips) * 1, circuit)
    loss.backward()
    
    print(loss)
    
def stim_mwpm():
    
    H = np.array([
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
    ])

    reps = 1
    code_sz = 3
    p = 1e-1
    n_shots = 10
    # sim = SurfaceCodeSim(reps, code_sz, p, n_shots)
    circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=reps,
            distance=code_sz,
            after_clifford_depolarization=p,
            after_reset_flip_probability=0,
            before_measure_flip_probability=p,
            before_round_data_depolarization=0,
        )
    # circuit = sim.get_circuit()
    det_coords = circuit.get_detector_coordinates()
    matching = pm.Matching.from_stim_circuit(circuit)
    print(det_coords)
    
    sampler = circuit.compile_detector_sampler()
    syndromes, flips = sampler.sample(n_shots, separate_observables=True)

    print(syndromes[0].shape)
    preds = matching.decode(syndromes[0])
    print(preds)
    print(matching)

    

    
if __name__ == "__main__":
    # stim_mwpm()
    main()    
    
        
        
        

