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
        self.epoch = saved_attributes["training_history"]["tot_epochs"] + 1
        self.model.load_state_dict(saved_attributes["model"])
        self.optimizer.load_state_dict(saved_attributes["optimizer"])
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
        return val_accuracy, bal_accuracy  #change to just bal i think
    
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
            _,top_accuracy = ls_inference(self.model,x, edge_index, edge_attr, batch_labels, detector_labels,flips)
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

            if self.save_model:
                self.save_model_w_training_settings()
                
            epoch_t = datetime.now() - start_t
            print(f"The epoch took: {epoch_t}, for {n_graphs} graphs.")

    def get_training_metrics(self):

        train_accuracy = self.training_history["train_accuracy"]
        val_accuracy = self.training_history["val_accuracy"]

        return train_accuracy, val_accuracy

