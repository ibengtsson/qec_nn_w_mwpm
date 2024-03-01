from typing import Any
import torch
import torch.nn as nn
import torch_geometric.nn as nng
from torch_geometric.utils import sort_edge_index
from scipy.spatial.distance import cdist
import numpy as np
#import pymatching
from qecsim.graphtools import mwpm
from src.graph import extract_edges
from signal import signal, SIGINT
import sys
from src.utils import inference

class LocalSearch:
    def __init__(self, model, search_radius, num_selections):
        self.model = model
        self.initial_score = torch.tensor(float(0))
        self.top_score = self.initial_score
        self.target = None
        self.vector = torch.nn.utils.parameters_to_vector(model.parameters())
        self.elite = self.vector.clone()
        self.n = self.vector.numel()
        self.running_idxs = np.arange(self.n)
        np.random.shuffle(self.running_idxs)
        self.idx = 0
        self.value = []  # list of indices
        self.num_selections = num_selections
        self.magnitude = torch.empty(self.num_selections,
                                dtype=self.vector.dtype,
                                device=self.vector.device)
        self.noise_vector = torch.empty_like(self.vector)
        self.jumped = False
        self.search_radius = search_radius

    def set_value(self):
        """Use the numpy choices function (which has no equivalent in Pytorch)
        to generate a sample from the array of indices. The sample size and
        distribution are dynamically updated by the algorithm's state.
        """
        self.check_idx()
        choices = self.running_idxs[self.idx:self.idx+self.num_selections]
        self.value = choices
        self.idx+=self.num_selections

    def check_idx(self):
        if (self.idx+self.num_selections)>self.n:
            self.idx=0
            np.random.shuffle(self.running_idxs)

    def set_noise(self):
        # Cast to precision and CUDA, and edit shape
        # search radius can be adjusted to fit scale of noise
        self.magnitude.uniform_(-self.search_radius, self.search_radius).squeeze()

    def set_noise_vector(self):
        """ This function defines a noise tensor, and returns it. The noise
        tensor needs to be the same shape as our originial vecotr. Hence, a
        "basis" tensor is created with zeros, then the chosen indices are
        modified.
        """
        self.noise_vector.fill_(0.)
        self.noise_vector[self.value] = self.magnitude

    def update_weights(self, model):
        nn.utils.vector_to_parameters(self.vector, model.parameters())

    def set_elite(self):
        self.jumped = False
        self.elite[self.value] = self.vector[self.value]
            #self.elite.clamp_(-0.9, 0.9)
            #self.elite.copy_(self.vector)
        self.jumped = True
            #self.frustration.reset_state()
        
    def set_vector(self):
        if not self.jumped:
            #self.vector.copy_(self.elite)
            elite_vals = self.elite[self.value]
            self.vector[self.value] = elite_vals

    def step(self,syndromes,flips):
        #print(self.vector)
        self.set_value()
        self.set_noise()
        self.set_noise_vector()
        self.vector[torch.from_numpy(self.value)] = self.vector[torch.from_numpy(self.value)] + torch.from_numpy(self.value)
        self.update_weights(self.model)
        _, new_accuracy = inference(self.model,syndromes,flips)
        if new_accuracy > self.top_score:
            self.set_elite()
            self.top_score = new_accuracy
        else:
            self.set_vector()
        self.idx += 1
            # decay to escape local maxima
            #self.top_score -= 0.002


    def return_topscore(self):
        return self.top_score