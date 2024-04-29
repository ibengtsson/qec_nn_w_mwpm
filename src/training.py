import torch
import numpy as np
import torch.nn as nn
import torch_geometric.nn as nng
import os
from pathlib import Path
from datetime import datetime
import random
from typing import Callable

from src.utils import parse_yaml, inference, ls_inference
from src.simulations import SurfaceCodeSim
from src.graph import get_batch_of_graphs
from src.local_search import LocalSearch

# new local search trainer for training with no epochs and split datasets
class LSTrainer_v2:
    def __init__(
        self,
        model: nn.Module,
        config: os.PathLike = None,
        save_model: bool = True, 
    ):
        # load and initialise settings
        paths, graph_settings, training_settings = parse_yaml(config)
        self.save_dir = Path(paths["save_dir"])
        self.saved_model_path = paths["saved_model_path"]
        self.graph_settings = graph_settings
        self.training_settings = training_settings
        self.save_model = save_model
        if "cuda" in training_settings["device"]:
            self.device = torch.device(
                training_settings["device"] if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device("cpu")

        # if not (
        #     self.device == torch.device("cuda") or self.device == torch.device("cpu")
        # ):
        #     torch.cuda.set_device(self.device)

        print(f"Running model on {self.device} with dataset size {self.training_settings['dataset_size']}.")
        # create a dictionary saving training metrics
        training_history = {}
        training_history["train_score"] = []
        training_history["alt_train_score"] = []
        training_history["val_score"] = []
        training_history["alt_val_score"] = []
        training_history["best_val_score"] = -1
        training_history["iter_improvement"] = []
        training_history["partial_time"] = []
        training_history["tot_time"] = 0
        self.training_history = training_history

        # move model to correct device
        self.model = model.to(self.device)

        # generate a unique name to not overwrite other models
        name = (
            "model_"
        )
        current_datetime = datetime.now().strftime("%y%m%d-%H%M%S")
        suffix = training_settings["suffix"]
        self.save_name = name + current_datetime + "_" + suffix
        # check if model should be loaded
        if training_settings["resume_training"]:
            self.load_trained_model()

    def save_model_w_training_settings(self, model_name=None):

        # make sure path exists, else create it
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if model_name is not None:
            path = self.save_dir / (model_name + ".pt")
        else:
            path = self.save_dir / (self.save_name + ".pt")
        self.optimal_weights = self.model.state_dict()
        if self.training_settings["resume_training"]:
            attributes = {
                "training_history": self.training_history,
                "model": self.optimal_weights,
                "model_vec": torch.nn.utils.parameters_to_vector(self.model.parameters()),
                "graph_settings": self.graph_settings,
                "training_settings": self.training_settings,
                "training_history_prev": self.training_history_prev
            }
        else:
            attributes = {
                "training_history": self.training_history,
                "model": self.optimal_weights,
                "model_vec": torch.nn.utils.parameters_to_vector(self.model.parameters()),
                "graph_settings": self.graph_settings,
                "training_settings": self.training_settings
            }
        torch.save(attributes, path)

    def load_trained_model(self):
        model_path = Path(self.saved_model_path)
        saved_attributes = torch.load(model_path, map_location=self.device)

        # update attributes and load model with trained weights
        self.training_history_prev = saved_attributes["training_history"]
        # older models do not have the attribute "best_val_accuracy"
        # if not "best_val_score" in self.training_history:
        #     self.training_history["best_val_score"] = -1
        self.model.load_state_dict(saved_attributes["model"])
        self.save_name = self.save_name + "_load_f_" + model_path.name.split(sep=".")[0]

        # only keep best found weights
        self.optimal_weights = saved_attributes["model"]

    # used for getting only one error probability sim
    def initialise_sim(self):
        # simulation settings
        code_size = self.graph_settings["code_size"]
        reps = self.graph_settings["repetitions"]
        dataset_size = self.training_settings["dataset_size"]
        p = self.graph_settings["one_error_rate"]

        task_dict = {
            "z": "surface_code:rotated_memory_z",
            "x": "surface_code:rotated_memory_x",
        }
        code_task = task_dict[self.graph_settings["experiment"]]

        sim = SurfaceCodeSim(
                reps,
                code_size,
                p,
                dataset_size,
                code_task=code_task,
            )

        return sim
    
    # used for getting multiple different error probabilities in data set
    def init_multi_error_p(self, n=5):  # check for comp with warmup
        # simulation settings
        code_size = self.graph_settings["code_size"]
        reps = self.graph_settings["repetitions"]
        dataset_size = self.training_settings["dataset_size"]

        min_error_rate = self.graph_settings["min_error_rate"]
        max_error_rate = self.graph_settings["max_error_rate"]

        error_rates = np.linspace(min_error_rate, max_error_rate, n)

        task_dict = {
            "z": "surface_code:rotated_memory_z",
            "x": "surface_code:rotated_memory_x",
        }
        code_task = task_dict[self.graph_settings["experiment"]]

        sims = []
        for p in error_rates:
            sim = SurfaceCodeSim(
                reps,
                code_size,
                p,
                dataset_size,
                code_task=code_task,
            )
            sims.append(sim)

        return sims
    
    # create a split test set used for validation
    def create_test_set(self, n_graphs=5e5, n=5):
        
        # simulation settings
        code_size = self.graph_settings["code_size"]
        reps = self.graph_settings["repetitions"]
        #min_error_rate = self.graph_settings["min_error_rate"]
        #max_error_rate = self.graph_settings["max_error_rate"]
        one_error_rate = self.graph_settings["one_error_rate"]

        #error_rates = np.linspace(min_error_rate, max_error_rate, n)

        task_dict = {
            "z": "surface_code:rotated_memory_z",
            "x": "surface_code:rotated_memory_x",
        }
        code_task = task_dict[self.graph_settings["experiment"]]

        #syndromes = []
        #flips = []
        #n_identities = 0
        #for p in error_rates:
        sim = SurfaceCodeSim(
                reps,
                code_size,
                one_error_rate,
                n_graphs,
                code_task=code_task,
            )
        syndromes, flips, n_identities = sim.generate_syndromes(use_for_mwpm=True)
            #syndromes.append(syndrome)
            #flips.append(flip)
            #n_identities += n_id

        #syndromes = np.concatenate(syndromes)
        #flips = np.concatenate(flips)

        # split into chunks to reduce memory footprint later
        batch_size = self.training_settings["batch_size_val"]
        n_splits = syndromes.shape[0] // batch_size + 1

        syndromes = np.array_split(syndromes, n_splits)
        flips = np.array_split(flips, n_splits)

        return syndromes, flips, n_identities
    
    # create a split set used for training with multiple different error prob
    def create_multi_p_set(self, n=5):
        
        # simulation settings
        code_size = self.graph_settings["code_size"]
        reps = self.graph_settings["repetitions"]
        min_error_rate = self.graph_settings["min_error_rate"]
        max_error_rate = self.graph_settings["max_error_rate"]
        dataset_size = self.training_settings["dataset_size"]
        error_rates = np.linspace(min_error_rate, max_error_rate, n)

        task_dict = {
            "z": "surface_code:rotated_memory_z",
            "x": "surface_code:rotated_memory_x",
        }
        code_task = task_dict[self.graph_settings["experiment"]]

        syndromes = []
        flips = []
        n_identities = 0
        for p in error_rates:
            sim = SurfaceCodeSim(
                reps,
                code_size,
                p,
                int(dataset_size / n),
                code_task=code_task,
            )
            syndrome, flip, n_id = sim.generate_syndromes(use_for_mwpm=True)
            syndromes.append(syndrome)
            flips.append(flip)
            n_identities += n_id

        syndromes = np.concatenate(syndromes)
        flips = np.concatenate(flips)

        # split into chunks to reduce memory footprint later
        batch_size = self.training_settings["batch_size_train"]
        n_splits = syndromes.shape[0] // batch_size + 1

        syndromes = np.array_split(syndromes, n_splits)
        flips = np.array_split(flips, n_splits)

        return syndromes, flips, n_identities
    
    # create a split set of syndromes used for training, one error prob only
    def create_split_train_set(self):
        
        # simulation settings
        code_size = self.graph_settings["code_size"]
        reps = self.graph_settings["repetitions"]
        one_error_rate = self.graph_settings["one_error_rate"]
        dataset_size = self.training_settings["dataset_size"]

        task_dict = {
            "z": "surface_code:rotated_memory_z",
            "x": "surface_code:rotated_memory_x",
        }
        code_task = task_dict[self.graph_settings["experiment"]]

        syndromes = []
        flips = []
        n_identities = 0
        sim = SurfaceCodeSim(
                reps,
                code_size,
                one_error_rate,
                dataset_size,
                code_task=code_task,
        )
        syndromes, flips, n_id = sim.generate_syndromes(use_for_mwpm=True)

        # split into chunks to reduce memory footprint later
        batch_size_train = self.training_settings["batch_size_train"] 
        n_splits = syndromes.shape[0] // batch_size_train + 1

        syndromes = np.array_split(syndromes, n_splits)
        flips = np.array_split(flips, n_splits)

        return syndromes, flips, n_identities
    
    def create_graph_set(self, syndromes, flips, n_identities):
        m_nearest_nodes = self.graph_settings["m_nearest_nodes"]
        experiment = self.graph_settings["experiment"]
        graph_sets = []
        for syndrome_set, flip_set in zip(syndromes, flips):
            x, edge_index, edge_attr, batch_labels, detector_labels = get_batch_of_graphs(
            syndrome_set, m_nearest_nodes, experiment=experiment, device=self.device
            )
            graph = {
                "x": x,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "batch_labels": batch_labels,
                "detector_labels": detector_labels,
                "flips": flip_set
            }
            graph_sets.append(graph)
        return graph_sets

    
    def evaluate_test_set(self, syndromes, flips, n_identities, n_graphs=5e4):
        # is n_graphs=n_syndromes? i dont think so
        m_nearest_nodes = self.graph_settings["m_nearest_nodes"]
        n_correct_preds = 0
        bal_acc = 0
        for syndrome, flip in zip(syndromes, flips):
            # this should maybe be changed to ls_inference and graphs generated before
            _n_correct_preds, _bal_acc = inference(
                self.model,
                syndrome,
                flip,
                experiment=self.graph_settings["experiment"],
                m_nearest_nodes=m_nearest_nodes,
                device=self.device,
            )
            n_correct_preds += _n_correct_preds
            bal_acc += _bal_acc
        val_accuracy = (n_correct_preds + n_identities) / n_graphs
        bal_accuracy = bal_acc/len(syndromes)
        return val_accuracy, bal_accuracy
    
    def evaluate_test_set_v2(self, val_graphs):
        n_correct_val = 0
        n_val_graphs = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for graph in val_graphs:
            x = graph["x"]
            edge_index = graph["edge_index"]
            edge_attr = graph["edge_attr"]
            batch_labels= graph["batch_labels"]
            detector_labels = graph["detector_labels"]
            flips = graph["flips"]
            _n_graphs = len(flips)
            n_val_graphs += _n_graphs
            _n_correct, _, _, tp_tn_fp_fn = ls_inference(self.model,x, edge_index, edge_attr, batch_labels, detector_labels,flips)
            n_correct_val += _n_correct
            TP += tp_tn_fp_fn[0]
            TN += tp_tn_fp_fn[1]
            FP += tp_tn_fp_fn[2]
            FN += tp_tn_fp_fn[3]
        accuracy = n_correct_val/n_val_graphs
        sens = TP/(TP+FN)
        spec = TN/(TN+FP)
        bal_acc = (sens+spec)/2
        return accuracy, bal_acc


    
    def train(self):
        tot_start_t = datetime.now()
        search_radius = self.training_settings["search_radius"]
        n_selections = self.training_settings["n_selections"]
        score_decay = self.training_settings["score_decay"]
        metric = self.training_settings["metric"]
        n_model_params = len(torch.nn.utils.parameters_to_vector(self.model.parameters()))
        n_dim_iter = n_model_params // n_selections

        # initialize local search model
        ls = LocalSearch(self.model, search_radius, n_selections, self.device, score_decay, metric)        

        # generate validation syndromes
        n_val_graphs = self.training_settings["validation_set_size"]
        val_syndromes, val_flips, n_val_identities = self.create_test_set(
            n_graphs=n_val_graphs,
        )
        # generate validation set graphs
        val_graphs = self.create_graph_set(val_syndromes, val_flips, n_val_identities)

        # set how many times all dimensions should be sampled
        repeat_selection = self.training_settings["repeat_selection"]
        if repeat_selection:
            n_repetitions = self.training_settings["n_repetitions"]
        else:
            n_repetitions = 1

        # create full dataset, split into smaller "batches"
        #syndromes, flips, n_trivial = self.create_multi_p_set()
        syndromes, flips, n_trivial = self.create_split_train_set()
        # create set of graphs from split syndrome set
        graph_set = self.create_graph_set(syndromes, flips, n_trivial)
        print("Number of dimension partitions:",n_dim_iter)
        partial_start_t = datetime.now()
        for i in range(n_dim_iter*n_repetitions):
                if self.training_settings["resume_training"] and i == 0:
                    accuracy, bal_acc = ls.run_inference(graph_set)
                    if metric is None or metric=="accuracy":
                        old_acc = accuracy
                    elif metric=="balanced":
                        old_acc = bal_acc
                else:
                    old_acc = ls.return_topscore()
                ls.step_split_data(graph_set)
                new_acc = ls.return_topscore()
                # we add new accuracy and i after each improvement
                if new_acc > old_acc:
                    print("New best found at iteration:",i)
                    print("New best accuracy:",new_acc)
                    alt_metric = ls.return_alternative_metric()
                    self.training_history["train_score"].append(new_acc)
                    self.training_history["iter_improvement"].append(i)
                    self.training_history["alt_train_score"].append(alt_metric)
                    # validation
                    val_accuracy, val_bal_acc = self.evaluate_test_set_v2(val_graphs)
                    print("Validation accuracy:",val_accuracy)
                    if metric == None or metric == "accuracy":
                        self.training_history["val_score"].append(val_accuracy)
                        self.training_history["alt_val_score"].append(val_bal_acc)
                        if val_accuracy > self.training_history["best_val_score"]:
                            self.training_history["best_val_score"] = val_accuracy
                    elif metric == "balanced":
                        self.training_history["val_score"].append(val_bal_acc)
                        self.training_history["alt_val_score"].append(val_accuracy)
                        if val_bal_acc > self.training_history["best_val_score"]:
                            self.training_history["best_val_score"] = val_bal_acc
                    partial_t = datetime.now() - partial_start_t
                    self.training_history["partial_time"].append(partial_t)
                    if self.save_model:
                        self.save_model_w_training_settings()
                elif i%500 == 0:
                    alt_metric = ls.return_alternative_metric()
                    self.training_history["train_score"].append(new_acc)
                    self.training_history["iter_improvement"].append(i)
                    self.training_history["alt_train_score"].append(alt_metric)
                    val_accuracy, val_bal_acc = self.evaluate_test_set_v2(val_graphs)
                    print("Validation accuracy:",val_accuracy)
                    if metric == None or metric == "accuracy":
                        self.training_history["val_score"].append(val_accuracy)
                        self.training_history["alt_val_score"].append(val_bal_acc)
                        if val_accuracy > self.training_history["best_val_score"]:
                            self.training_history["best_val_score"] = val_accuracy
                    elif metric == "balanced":
                        self.training_history["val_score"].append(val_bal_acc)
                        self.training_history["alt_val_score"].append(val_accuracy)
                        if val_bal_acc > self.training_history["best_val_score"]:
                            self.training_history["best_val_score"] = val_bal_acc
                    if self.save_model:
                        self.save_model_w_training_settings()         
        tot_t = datetime.now() - tot_start_t
        self.training_history["tot_time"]=tot_t
        if self.save_model:
            self.save_model_w_training_settings()
        print(f"The training took: {tot_t}")

    def get_training_metrics(self):

        train_score = self.training_history["train_score"]
        #alt_train_score = self.training_history["alt_train_score"]
        val_score = self.training_history["val_score"]
        #alt_val_score = self.training_history["alt_val_score"]
        #acc = self.training_history["comb_accuracy"]
        time = self.training_history["partial_time"]

        return train_score, val_score, time

               

class LSTrainer:

    def __init__(
        self,
        model: nn.Module,
        config: os.PathLike = None,
        save_model: bool = True,
    ):

        # load and initialise settings
        paths, graph_settings, training_settings = parse_yaml(config)
        self.save_dir = Path(paths["save_dir"])
        self.saved_model_path = paths["saved_model_path"]
        self.graph_settings = graph_settings
        self.training_settings = training_settings
        self.save_model = save_model
        
        
        self.accuracy = []

        # current training status
        self.warmup_epochs = training_settings["warmup_epochs"]
        self.epoch = training_settings["current_epoch"]
        if "cuda" in training_settings["device"]:
            self.device = torch.device(
                training_settings["device"] if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device("cpu")

        print(f"Running model on {self.device} with dataset size {self.training_settings['dataset_size']}.")
        # create a dictionary saving training metrics
        training_history = {}
        training_history["epoch"] = self.epoch  # this will need to be adjusted to fit warmup+train
        training_history["warmup_train_loss"] = []
        training_history["train_accuracy"] = []
        #training_history["train_loss"] = []
        training_history["val_accuracy"] = []
        training_history["best_val_accuracy"] = -1
        training_history["iter_improvement"] = []
        training_history["comb_accuracy"] = []

        self.training_history = training_history

        # only keep best found weights
        self.optimal_weights = None

        # move model to correct device, save loss and instantiate the optimizer
        self.model = model.to(self.device)
        # optimizer only used for warmup training
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=training_settings["warmup_lr"]
        )   # check this, if it can only be used in warmup

        # generate a unique name to not overwrite other models
        name = (
            "d"
            + str(graph_settings["code_size"])
            + "_d_t_"
            + str(graph_settings["repetitions"])
            + "_"
        )
        current_datetime = datetime.now().strftime("%y%m%d-%H%M%S")
        self.save_name = name + current_datetime

        # check if model should be loaded
        if training_settings["resume_training"]:
            self.load_trained_model()

    def save_model_w_training_settings(self, model_name=None):

        # make sure path exists, else create it
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if model_name is not None:
            path = self.save_dir / (model_name + ".pt")
        else:
            path = self.save_dir / (self.save_name + ".pt")

        # we only want to save the weights that corresponds to the best found accuracy
        if (
            self.training_history["val_accuracy"][-1]
            > self.training_history["best_val_accuracy"]
        ):
            self.training_history["best_val_accuracy"] = self.training_history[
                "val_accuracy"
            ][-1]
            self.optimal_weights = self.model.state_dict()

        attributes = {
            "training_history": self.training_history,
            "model": self.optimal_weights,
            "optimizer": self.optimizer.state_dict(),
            "graph_settings": self.graph_settings,
            "training_settings": self.training_settings,
        }

        torch.save(attributes, path)

    def load_trained_model(self):
        model_path = Path(self.saved_model_path)
        saved_attributes = torch.load(model_path, map_location=self.device)

        # update attributes and load model with trained weights
        self.training_history = saved_attributes["training_history"]

        # older models do not have the attribute "best_val_accuracy"
        if not "best_val_accuracy" in self.training_history:
            self.training_history["best_val_accuracy"] = -1
        if not "train_accuracy" in self.training_history:
            self.training_history["train_accuracy"] = []
        if not "iter_improvement" in self.training_history:
            self.training_history["iter_improvement"] = []
        if not "comb_accuracy" in self.training_history:
            self.training_history["comb_accuracy"] = []
        #self.epoch = saved_attributes["training_history"]["tot_epochs"] + 1
        self.model.load_state_dict(saved_attributes["model"])
        #self.optimizer.load_state_dict(saved_attributes["optimizer"])
        self.save_name = self.save_name + "_load_f_" + model_path.name.split(sep=".")[0]

        # only keep best found weights
        self.optimal_weights = saved_attributes["model"]


    def initialise_simulations(self, n=5):  # check for comp with warmup
        # simulation settings
        code_size = self.graph_settings["code_size"]
        reps = self.graph_settings["repetitions"]
        dataset_size = self.training_settings["dataset_size"]

        min_error_rate = self.graph_settings["min_error_rate"]
        max_error_rate = self.graph_settings["max_error_rate"]

        error_rates = np.linspace(min_error_rate, max_error_rate, n)

        task_dict = {
            "z": "surface_code:rotated_memory_z",
            "x": "surface_code:rotated_memory_x",
        }
        code_task = task_dict[self.graph_settings["experiment"]]

        sims = []
        for p in error_rates:
            sim = SurfaceCodeSim(
                reps,
                code_size,
                p,
                dataset_size,
                code_task=code_task,
            )
            sims.append(sim)

        return sims
    
    # used for getting only one error probability sim
    def initialise_sim(self):
        # simulation settings
        code_size = self.graph_settings["code_size"]
        reps = self.graph_settings["repetitions"]
        dataset_size = self.training_settings["dataset_size"]
        p = self.graph_settings["one_error_rate"]

        task_dict = {
            "z": "surface_code:rotated_memory_z",
            "x": "surface_code:rotated_memory_x",
        }
        code_task = task_dict[self.graph_settings["experiment"]]

        sim = SurfaceCodeSim(
                reps,
                code_size,
                p,
                dataset_size,
                code_task=code_task,
            )

        return sim
    

    def create_test_set(self, n_graphs=5e5, n=5):   # this is only used for validation atm
        
        # simulation settings
        code_size = self.graph_settings["code_size"]
        reps = self.graph_settings["repetitions"]
        min_error_rate = self.graph_settings["min_error_rate"]
        max_error_rate = self.graph_settings["max_error_rate"]

        error_rates = np.linspace(min_error_rate, max_error_rate, n)

        task_dict = {
            "z": "surface_code:rotated_memory_z",
            "x": "surface_code:rotated_memory_x",
        }
        code_task = task_dict[self.graph_settings["experiment"]]

        syndromes = []
        flips = []
        n_identities = 0
        for p in error_rates:
            sim = SurfaceCodeSim(
                reps,
                code_size,
                p,
                int(n_graphs / n),
                code_task=code_task,
            )
            syndrome, flip, n_id = sim.generate_syndromes(use_for_mwpm=True)
            syndromes.append(syndrome)
            flips.append(flip)
            n_identities += n_id

        syndromes = np.concatenate(syndromes)
        flips = np.concatenate(flips)

        # split into chunks to reduce memory footprint later
        batch_size = self.training_settings["batch_size"]   # maybe change?
        n_splits = syndromes.shape[0] // batch_size + 1

        syndromes = np.array_split(syndromes, n_splits)
        flips = np.array_split(flips, n_splits)

        return syndromes, flips, n_identities
    
    def evaluate_test_set(self, syndromes, flips, n_identities, n_graphs=5e4):
        # is n_graphs=n_syndromes? i dont think so
        m_nearest_nodes = self.graph_settings["m_nearest_nodes"]
        n_correct_preds = 0
        bal_acc = 0
        for syndrome, flip in zip(syndromes, flips):
            # this should maybe be changed to ls_inference and graphs generated before
            _n_correct_preds, _bal_acc = inference(
                self.model,
                syndrome,
                flip,
                experiment=self.graph_settings["experiment"],
                m_nearest_nodes=m_nearest_nodes,
                device=self.device,
            )
            n_correct_preds += _n_correct_preds
            bal_acc += _bal_acc
        val_accuracy = (n_correct_preds + n_identities) / n_graphs
        bal_accuracy = bal_acc/len(syndromes)
        return val_accuracy, bal_accuracy
    
    def train_warmup(self):

        # training settings
        current_epoch = self.epoch
        dataset_size = self.training_settings["dataset_size"]
        batch_size = self.training_settings["batch_size"]
        n_batches = dataset_size // batch_size
        #gradient_factor = self.training_settings["gradient_factor"]

        loss_fun = nn.MSELoss()
        n_epochs = self.training_settings["warmup_epochs"]


        # initialise simulations and graph settings
        m_nearest_nodes = self.graph_settings["m_nearest_nodes"]
        n_node_features = self.graph_settings["n_node_features"]
        power = self.graph_settings["power"]
        # create warmup dataset? with batch size = epoch dataset size for normal train
        sims = self.initialise_simulations()

        # generate validation syndromes
        n_val_graphs = self.training_settings["validation_set_size"]
        val_syndromes, val_flips, n_val_identities = self.create_test_set(
            n_graphs=n_val_graphs,
        )
        for epoch in range(current_epoch, n_epochs):
            train_loss = 0
            epoch_n_graphs = 0
            epoch_n_trivial = 0
            print(f"Epoch {epoch}")
            start_t = datetime.now()
            
            for _ in range(n_batches):
                # simulate data as we go
                sim = random.choice(sims)
                syndromes, flips, n_trivial = sim.generate_syndromes(use_for_mwpm=True)
                epoch_n_trivial += n_trivial
                x, edge_index, edge_attr, batch_labels, detector_labels = (
                    get_batch_of_graphs(
                        syndromes,
                        m_nearest_nodes,
                        n_node_features=n_node_features,
                        power=power,
                        device=self.device,
                    )
                )

                n_graphs = syndromes.shape[0]

                # forward/backward pass
                self.optimizer.zero_grad()
                edge_weights, label = self.model(
                        x,
                        edge_index,
                        edge_attr,
                        detector_labels,
                        batch_labels,
                        warmup=True,
                    )
                loss = loss_fun(edge_weights, label)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * n_graphs
                epoch_n_graphs += n_graphs

            # compute losses and logical accuracy
            # ------------------------------------

            # train
            train_loss /= epoch_n_graphs

            # validation
            val_accuracy, bal_accuracy = self.evaluate_test_set(
                val_syndromes,
                val_flips,
                n_val_identities,
                n_graphs=n_val_graphs,
            )

            # save training attributes after every epoch
            self.epoch = epoch
            self.training_history["epoch"] = epoch
            self.training_history["warmup_train_loss"].append(train_loss)
            self.training_history["val_accuracy"].append(bal_accuracy)

            if self.save_model:
                self.save_model_w_training_settings()
                
            epoch_t = datetime.now() - start_t
            print(f"The warmup epoch took: {epoch_t}, for {n_graphs} graphs.")
        self.epoch += 1

    def train(self):

        # training settings
        current_epoch = self.epoch
        #dataset_size = self.training_settings["dataset_size"]
        #n_batches = dataset_size // batch_size

        n_epochs = self.training_settings["tot_epochs"]
        search_radius = self.training_settings["search_radius"]
        n_selections = self.training_settings["n_selections"]
        experiment = self.graph_settings["experiment"]
        n_model_params = len(torch.nn.utils.parameters_to_vector(self.model.parameters()))
        n_dim_iter = n_model_params // n_selections
        # training optimizer
        ls = LocalSearch(self.model, search_radius, n_selections, self.device)        

        # initialise simulations and graph settings
        m_nearest_nodes = self.graph_settings["m_nearest_nodes"]

        sims = self.initialise_simulations()

        # generate validation syndromes
        n_val_graphs = self.training_settings["validation_set_size"]
        val_syndromes, val_flips, n_val_identities = self.create_test_set(
            n_graphs=n_val_graphs,
        )
        for epoch in range(current_epoch, n_epochs):
            print(f"Epoch {epoch}")
            sim = random.choice(sims)
            syndromes, flips, n_trivial = sim.generate_syndromes(use_for_mwpm=True)
            epoch_n_trivial = n_trivial
            n_graphs = syndromes.shape[0]
            start_t = datetime.now()
            x, edge_index, edge_attr, batch_labels, detector_labels = get_batch_of_graphs(
            syndromes, m_nearest_nodes, experiment=experiment, device=self.device
            )
            _,top_accuracy, accuracy = ls_inference(self.model,x, edge_index, edge_attr, batch_labels, detector_labels,flips)
            ls.top_score = top_accuracy
            for i in range(n_dim_iter):
                ls.step(x, edge_index, edge_attr, batch_labels, detector_labels,flips)

            # update model to best version after local search
            nn.utils.vector_to_parameters(ls.elite, self.model.parameters())

            # validation
            val_accuracy, bal_accuracy = self.evaluate_test_set(
                val_syndromes,
                val_flips,
                n_val_identities,
                n_graphs=n_val_graphs,
            )

            # save training attributes after every epoch
            self.epoch = epoch
            self.training_history["epoch"] = epoch
            self.training_history["train_accuracy"].append(ls.top_score)
            self.training_history["val_accuracy"].append(bal_accuracy)
            self.accuracy.append(accuracy)

            if self.save_model:
                self.save_model_w_training_settings()
                
            epoch_t = datetime.now() - start_t
            print(f"The epoch took: {epoch_t}, for {n_graphs} graphs.")


    def train_v2(self):
        dataset_size = self.training_settings["dataset_size"]
        search_radius = self.training_settings["search_radius"]
        n_selections = self.training_settings["n_selections"]
        experiment = self.graph_settings["experiment"]
        n_model_params = len(torch.nn.utils.parameters_to_vector(self.model.parameters()))
        n_dim_iter = n_model_params // n_selections
        # training optimizer
        ls = LocalSearch(self.model, search_radius, n_selections, self.device)        

        # initialise simulations and graph settings
        m_nearest_nodes = self.graph_settings["m_nearest_nodes"]

        sim = self.initialise_sim()

        # generate validation syndromes
        n_val_graphs = self.training_settings["validation_set_size"]
        val_syndromes, val_flips, n_val_identities = self.create_test_set(
            n_graphs=n_val_graphs,
        )
        repeat_selection = self.training_settings["repeat_selection"]
        if repeat_selection:
            n_repetitions = self.training_settings["n_repetitions"]
        else:
            n_repetitions = 1
        syndromes, flips, n_trivial = sim.generate_syndromes(use_for_mwpm=True)
        n_graphs = syndromes.shape[0]
        start_t = datetime.now()
        x, edge_index, edge_attr, batch_labels, detector_labels = get_batch_of_graphs(
            syndromes, m_nearest_nodes, experiment=experiment, device=self.device
            )
        _,top_accuracy, accuracy = ls_inference(self.model,x, edge_index, edge_attr, batch_labels, detector_labels,flips)
        ls.top_score = top_accuracy
        self.training_history["train_accuracy"].append(ls.top_score)
        self.training_history["iter_improvement"].append(0)
        print("Number of dimension partitions:",n_dim_iter)
        for i in range(n_dim_iter*n_repetitions):
                old_acc = ls.top_score
                ls.step(x, edge_index, edge_attr, batch_labels, detector_labels,flips)
                new_acc = ls.top_score
                # we add new accuracy and i after each improvement
                if ~np.equal(new_acc, old_acc):
                    print("New best found at iteration:",i)
                    print("New best accuracy:",new_acc)
                    self.training_history["train_accuracy"].append(ls.top_score)
                    self.training_history["iter_improvement"].append(i+1)
                    self.training_history["comb_accuracy"].append(ls.accuracy)
                    # validation
                    val_accuracy, bal_accuracy = self.evaluate_test_set(
                    val_syndromes,
                    val_flips,
                    n_val_identities,
                    n_graphs=n_val_graphs,
                    )
                    self.training_history["val_accuracy"].append(bal_accuracy)
                    if self.save_model:
                        self.save_model_w_training_settings()

        # update model to best version after local search
        # nn.utils.vector_to_parameters(ls.elite, self.model.parameters())
               
        epoch_t = datetime.now() - start_t
        print(f"The training took: {epoch_t}, for {n_graphs} graphs.")

    # def train_split_dataset(self):
    #     dataset_size = self.training_settings["dataset_size"]
    #     batch_train = self.training_settings["batch_train"]
    #     search_radius = self.training_settings["search_radius"]
    #     n_selections = self.training_settings["n_selections"]
    #     experiment = self.graph_settings["experiment"]
    #     n_model_params = len(torch.nn.utils.parameters_to_vector(self.model.parameters()))
    #     n_dim_iter = n_model_params // n_selections
    #     # training optimizer
    #     ls = LocalSearch(self.model, search_radius, n_selections, self.device)        

    #     # initialise simulations and graph settings
    #     m_nearest_nodes = self.graph_settings["m_nearest_nodes"]

    #     sim = self.initialise_sim()
    #     # generate validation syndromes
    #     n_val_graphs = self.training_settings["validation_set_size"]
    #     val_syndromes, val_flips, n_val_identities = self.create_test_set(
    #         n_graphs=n_val_graphs,
    #     )
    #     repeat_selection = self.training_settings["repeat_selection"]
    #     if repeat_selection:
    #         n_repetitions = self.training_settings["n_repetitions"]
    #     else:
    #         n_repetitions = 1
    #     syndromes, flips, n_trivial = sim.generate_syndromes(use_for_mwpm=True)
    #     # split into chunks to reduce memory footprint later
    #     n_splits = syndromes.shape[0] // batch_train + 1
    #     syndromes = np.array_split(syndromes, n_splits)
    #     flips = np.array_split(flips, n_splits)
    #     start_t = datetime.now()
    #     graphs = []
    #     n_correct = 0
    #     n_graphs = 0
    #     for syndrome, flip in zip(syndromes, flips):
    #         _n_graphs = syndrome.shape[0]
    #         n_graphs += _n_graphs
    #         x, edge_index, edge_attr, batch_labels, detector_labels = get_batch_of_graphs(
    #         syndrome, m_nearest_nodes, experiment=experiment, device=self.device
    #         )
    #         graph = {"x":x, "edge_index":edge_index, "edge_attr":edge_attr, "batch_labels":batch_labels, "detector_labels":detector_labels, "flips":flip}
    #         graphs.append(graph)
    #         _n_correct ,top_accuracy, accuracy = ls_inference(self.model,x, edge_index, edge_attr, batch_labels, detector_labels,flip)
    #         n_correct += _n_correct
            
    #     ls.top_score = n_correct/n_graphs
    #     self.training_history["train_accuracy"].append(ls.top_score)
    #     self.training_history["iter_improvement"].append(0)
    #     print("Number of dimension partitions:",n_dim_iter)
    #     for i in range(n_dim_iter*n_repetitions):
    #             old_acc = ls.top_score
    #             ls.step(x, edge_index, edge_attr, batch_labels, detector_labels,flips)
    #             new_acc = ls.top_score
    #             # we add new accuracy and i after each improvement
    #             if ~np.equal(new_acc, old_acc):
    #                 print("New best found at iteration:",i)
    #                 print("New best accuracy:",new_acc)
    #                 self.training_history["train_accuracy"].append(ls.top_score)
    #                 self.training_history["iter_improvement"].append(i+1)
    #                 self.training_history["comb_accuracy"].append(ls.accuracy)
    #                 # validation
    #                 val_accuracy, bal_accuracy = self.evaluate_test_set(
    #                 val_syndromes,
    #                 val_flips,
    #                 n_val_identities,
    #                 n_graphs=n_val_graphs,
    #                 )
    #                 self.training_history["val_accuracy"].append(bal_accuracy)
    #                 if self.save_model:
    #                     self.save_model_w_training_settings()

    #     # update model to best version after local search
    #     # nn.utils.vector_to_parameters(ls.elite, self.model.parameters())
               
    #     epoch_t = datetime.now() - start_t
    #     print(f"The training took: {epoch_t}, for {n_graphs} graphs.")


    def get_training_metrics(self):

        train_accuracy = self.training_history["train_accuracy"]
        val_accuracy = self.training_history["val_accuracy"]
        acc = self.training_history["comb_accuracy"]

        return train_accuracy, val_accuracy, acc

