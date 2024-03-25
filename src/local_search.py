import torch
import torch.nn as nn
import numpy as np
from src.utils import inference, ls_inference

class LocalSearch:
    def __init__(
            self, 
            model, 
            search_radius, 
            num_selections, 
            device, 
            score_decay=None,
            metric = None):
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
        self.alt_score = 0
        self.score_decay = score_decay
        self.metric = metric

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

    def run_inference(self, graphs):
        n_correct = 0
        n_graphs = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for graph in graphs:
            x = graph["x"]
            edge_index = graph["edge_index"]
            edge_attr = graph["edge_attr"]
            batch_labels = graph["batch_labels"]
            detector_labels = graph["detector_labels"]
            flips = graph["flips"]
            _n_graphs = len(flips)
            _n_correct, _, _, tp_tn_fp_fn = ls_inference(self.model,x, edge_index, edge_attr, batch_labels, detector_labels,flips)
            n_correct += _n_correct
            n_graphs += _n_graphs
            TP += tp_tn_fp_fn[0]
            TN += tp_tn_fp_fn[1]
            FP += tp_tn_fp_fn[2]
            FN += tp_tn_fp_fn[3]

        accuracy = n_correct/n_graphs
        sens = TP/(TP+FN)
        spec = TN/(TN+FP)
        bal_acc = (sens+spec)/2
        return accuracy, bal_acc

    def step(self,x, edge_index, edge_attr, batch_labels, detector_labels, flips):
        #print(self.vector)
        self.set_value()
        self.set_noise()
        #self.set_noise_vector()

        self.vector[self.value] = self.vector[self.value] + self.magnitude
        self.update_weights(self.model)
        _, bal_acc, accuracy, _ = ls_inference(self.model,x, edge_index, edge_attr, batch_labels, detector_labels, flips)
        if self.metric is None or self.metric == "accuracy":
            used_score = accuracy
            alt_score = bal_acc
        elif self.metric == "balanced":
            used_score = bal_acc
            alt_score = accuracy
        else:
            print(f"Metric {self.metric} is not defined")
        if used_score > self.top_score:
            self.set_elite()
            self.top_score = used_score
            self.alt_score = alt_score
        else:
            self.set_vector()
        self.idx += 1
        # decay to escape local maxima
        if self.score_decay is not None:
            self.top_score -= self.score_decay

    def step_split_data(self, graphs):
        #print(self.vector)
        self.set_value()
        self.set_noise()
        #self.set_noise_vector()
        self.vector[self.value] = self.vector[self.value] + self.magnitude
        self.update_weights(self.model)
        accuracy, bal_acc = self.run_inference(graphs)
        if self.metric is None or self.metric == "accuracy":
            used_score = accuracy
            alt_score = bal_acc
        elif self.metric == "balanced":
            used_score = bal_acc
            alt_score = accuracy
        else:
            print(f"Metric {self.metric} not defined")
        if used_score > self.top_score:
            self.set_elite()
            self.top_score = used_score
            self.alt_score = alt_score
        else:
            self.set_vector()
        self.idx += 1
        # decay to escape local maxima
        if self.score_decay is not None:
            self.top_score -= self.score_decay


    def return_topscore(self):
        return self.top_score
    
    def return_alternative_metric(self):
        return self.alt_score
