import torch
import numpy as np
import torch.nn as nn
import torch_geometric.nn as nng
import os
import sys
from pathlib import Path
from datetime import datetime

# relative imports
sys.path.append("../")
from src.utils import parse_yaml
from src.simulations import SurfaceCodeSim


class ModelTrainer:

    def __init__(
        self,
        model: nn.Module,
        loss_fun,
        config: os.PathLike = None,
    ):

        # load and initialise settings
        paths, graph_settings, training_settings = parse_yaml(config)
        self.save_dir = Path(paths["save_dir"])
        self.saved_model_path = Path(paths["saved_model_path"])
        self.graph_settings = graph_settings
        self.training_settings = training_settings
        
        # current training status
        self.epoch = training_settings["current_epoch"]
        if training_settings["device"] == "cuda":
            self.device = torch.device(
                training_settings["device"] if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device("cpu")
            
        # create a dictionary saving training metrics
        training_history = {}
        training_history["epoch"] = self.epoch
        training_history["train_accuracy"] = []
        training_history["val_accuracy"] = []
        training_history["best_val_accuracy"] = -1

        self.training_history = training_history
        
        # only keep best found weights
        self.optimal_weights = None

        # move model to correct device, save loss and instantiate the optimizer
        self.model = model.to(self.device)
        self.loss_fun = loss_fun
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=training_settings["lr"]
        )
        
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
        if self.training_history["val_accuracy"][-1] > self.training_history["best_val_accuracy"]:
            self.training_history["best_val_accuracy"] = self.training_history["val_accuracy"][-1]
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
        self.epoch = saved_attributes["training_history"]["epoch"] + 1
        self.model.load_state_dict(saved_attributes["model"])
        self.optimizer.load_state_dict(saved_attributes["optimizer"])
        self.save_name = self.save_name + "_load_f_" + model_path.name.split(sep=".")[0]
        
        # only keep best found weights
        self.optimal_weights = saved_attributes["model"]
        
    def initialise_simulations(self, n=5):
        # simulation settings
        code_size = self.graph_settings["code_size"]
        reps = self.graph_settings["repetitions"]
        batch_size = self.training_settings["batch_size"]

        min_error_rate = self.graph_settings["min_error_rate"]
        max_error_rate = self.graph_settings["max_error_rate"]

        error_rates = np.linspace(min_error_rate, max_error_rate, n)

        sims = []
        for p in error_rates:
            sim = SurfaceCodeSim(reps, code_size, p, batch_size)
            sims.append(sim)

        return sims

    def create_test_set(self, n_graphs=5e5, n=5):
        # simulation settings
        code_size = self.graph_settings["code_size"]
        reps = self.graph_settings["repetitions"]
        min_error_rate = self.graph_settings["min_error_rate"]
        max_error_rate = self.graph_settings["max_error_rate"]

        error_rates = np.linspace(min_error_rate, max_error_rate, n)

        syndromes = []
        flips = []
        n_identities = 0
        for p in error_rates:
            sim = SurfaceCodeSim(reps, code_size, p, int(n_graphs / n))
            syndrome, flip, n_id = sim.generate_syndromes()
            syndromes.append(syndrome)
            flips.append(flip)
            n_identities += n_id

        syndromes = np.concatenate(syndromes)
        flips = np.concatenate(flips)

        # split into chunks to reduce memory footprint later
        batch_size = self.training_settings["batch_size"]
        n_splits = syndromes.shape[0] // batch_size + 1

        syndromes = np.array_split(syndromes, n_splits)
        flips = np.array_split(flips, n_splits)

        return syndromes, flips, n_identities