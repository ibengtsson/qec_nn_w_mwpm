import torch
import torch.nn as nn
import numpy as np
from src.utils import inference, ls_inference

class LocalSearch:
    def __init__(self, model, search_radius, num_selections, device):
        self.model = model
        self.device = device
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
        self.accuracy = 0

    def set_value(self):
        """Use the numpy choices function (which has no equivalent in Pytorch)
        to generate a sample from the array of indices. The sample size and
        distribution are dynamically updated by the algorithm's state.
        """
        self.check_idx()
        choices = self.running_idxs[self.idx:self.idx+self.num_selections]
        self.value = torch.from_numpy(choices)
        self.value = self.value.to(self.device)
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

    def step(self,x, edge_index, edge_attr, batch_labels, detector_labels,flips):
        #print(self.vector)
        self.set_value()
        self.set_noise()
        #self.set_noise_vector()

        self.vector[self.value] = self.vector[self.value] + self.magnitude
        self.update_weights(self.model)
        _, new_accuracy, accuracy = ls_inference(self.model,x, edge_index, edge_attr, batch_labels, detector_labels,flips)
        if new_accuracy > self.top_score:
            self.set_elite()
            self.top_score = new_accuracy
            self.accuracy = accuracy
        else:
            self.set_vector()
        self.idx += 1
        #self.update_weights(self.model)
            # decay to escape local maxima
            #self.top_score -= 0.002

    # def step_test(self, x, t, loss_fcn):
    #     self.set_value()
    #     self.set_noise()
    #     self.vector[self.value] = self.vector[self.value] + self.magnitude
    #     self.update_weights(self.model)
    #     new_pred = self.model(x)
    #     loss = loss_fcn(new_pred, t)
    #     if loss < self.top_score:
    #         self.set_elite()
    #         self.top_score = loss
    #     else:
    #         self.set_vector()
    #     self.idx += 1
    #     if loss < 1000:
    #         self.search_radius = 0.01

    def step_split_data(self, graphs):
        #print(self.vector)
        self.set_value()
        self.set_noise()
        #self.set_noise_vector()
        self.vector[self.value] = self.vector[self.value] + self.magnitude
        self.update_weights(self.model)
        n_correct = 0
        n_graphs = 0
        for graph in graphs:
            x = graph["x"]
            edge_index = graph["edge_index"]
            edge_attr = graph["edge_attr"]
            batch_labels = graph["batch_labels"]
            detector_labels = graph["detector_labels"]
            flips = graph["flips"]
            _n_graphs = len(flips)
            _n_correct, new_accuracy, _ = ls_inference(self.model,x, edge_index, edge_attr, batch_labels, detector_labels,flips)
            n_correct += _n_correct
            n_graphs += _n_graphs

        new_accuracy = n_correct/n_graphs
        if new_accuracy > self.top_score:
            self.set_elite()
            self.top_score = new_accuracy
            #self.accuracy = accuracy
        else:
            self.set_vector()
        self.idx += 1
        #self.update_weights(self.model)
            # decay to escape local maxima
            #self.top_score -= 0.002


    def return_topscore(self):
        return self.top_score
