import torch
import numpy as np
import torch.nn as nn
import torch_geometric.nn as nng
import os
from pathlib import Path
from datetime import datetime
import random
import pandas as pd

from src.utils import parse_yaml, inference, predict_mwpm_nested
from src.simulations import SurfaceCodeSim
from src.graph import get_batch_of_graphs
from src.models import GraphNNV2
from src.losses import NestedMWPMLoss


class NestedModelTrainer:

    def __init__(
        self,
        config: os.PathLike = None,
        save_model: bool = True,
        seeds: bool = False,
    ):
        # set seed if desired
        if seeds:
            random.seed(747)
            torch.manual_seed(747)
            self.simul_seed = 747
        else:
            self.simul_seed = None
            

        # load and initialise settings
        paths, graph_settings, model_settings, training_settings = parse_yaml(config)
        self.save_dir = Path(paths["save_dir"])
        self.saved_model_path = paths["saved_model_path"]
        self.graph_settings = graph_settings
        self.model_settings = model_settings
        self.training_settings = training_settings
        self.save_model = save_model
        
        # current training status
        self.epoch = training_settings["current_epoch"]
        if "cuda" in training_settings["device"]:
            self.device = torch.device(
                training_settings["device"] if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device("cpu")

        print(
            f"Running model on {self.device} with batch size {training_settings['batch_size']}."
        )
        # create a dictionary saving training metrics
        training_history = {}
        training_history["epoch"] = self.epoch
        training_history["train_loss"] = []
        training_history["val_accuracy"] = []
        training_history["best_val_accuracy"] = -1

        self.training_history = training_history

        # only keep best found weights
        self.optimal_weights = None

        # move model to correct device, save loss and instantiate the optimizer
        if not (
            self.device == torch.device("cuda") or self.device == torch.device("cpu")
        ):
            torch.cuda.set_device(self.device)

        self.model = GraphNNV2(
            hidden_channels_GCN=model_settings["hidden_channels_GCN"],
            hidden_channels_MLP=model_settings["hidden_channels_MLP"],
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=training_settings["lr"],
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
        name = name + current_datetime
        save_path = Path(paths["save_dir"]) / (name + ".pt")
        
        # make sure we did not create an existing name
        if save_path.is_file():
            save_path = Path(paths["save_dir"]) / (name + "_1.pt")
        self.save_path = save_path

        # check if model should be loaded
        if training_settings["resume_training"]:
            self.load_trained_model()

    def save_model_w_training_settings(self):

        # make sure the save folder exists, else create it
        self.save_dir.mkdir(parents=True, exist_ok=True)

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

        torch.save(attributes, self.save_path)

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

    def initialise_simulations(self, n=5):
        
        # simulation settings
        code_size = self.graph_settings["code_size"]
        reps = self.graph_settings["repetitions"]
        batch_size = self.training_settings["batch_size"]

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
                batch_size,
                code_task=code_task,
            )
            sims.append(sim)

        return sims

    def create_test_set(self, error_rate=None, n_graphs=5e5, n=5):

        # simulation settings
        code_size = self.graph_settings["code_size"]
        reps = self.graph_settings["repetitions"]
        
        if error_rate is None:
            min_error_rate = self.graph_settings["min_error_rate"]
            max_error_rate = self.graph_settings["max_error_rate"]
        else:
            min_error_rate = max_error_rate = error_rate
            n = 1

        error_rates = np.linspace(min_error_rate, max_error_rate, n)

        task_dict = {
            "z": "surface_code:rotated_memory_z",
            "x": "surface_code:rotated_memory_x",
        }
        code_task = task_dict[self.graph_settings["experiment"]]

        syndromes = []
        flips = []
        n_identities = 0
        
         # initalise simulations with different error rates
        for i, p in enumerate(error_rates):
            
            # potential seed
            if self.simul_seed:
                seed = self.simul_seed + i
            else:
                seed = None

            sim = SurfaceCodeSim(
                reps,
                code_size,
                p,
                int(n_graphs / n),
                code_task=code_task,
            )
            
            syndrome, flip, n_id = sim.generate_syndromes(use_for_mwpm=True, seed=seed)
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

    def evaluate_test_set(self, syndromes, flips, n_identities, n_graphs=5e4):

        m_nearest_nodes = self.graph_settings["m_nearest_nodes"]
        n_correct_preds = 0
        n_syndromes = 0
        for syndrome, flip in zip(syndromes, flips):

            n_syndromes += syndrome.shape[0]
            _n_correct_preds = inference(
                self.model,
                syndrome,
                flip,
                experiment=self.graph_settings["experiment"],
                m_nearest_nodes=m_nearest_nodes,
                device=self.device,
                nested_tensors=True,
            )
            n_correct_preds += _n_correct_preds
        val_accuracy = (n_correct_preds) / n_syndromes
        # val_accuracy = (n_correct_preds + n_identities) / n_graphs

        return val_accuracy

    def train(self):

        # training settings
        current_epoch = self.epoch
        dataset_size = self.training_settings["dataset_size"]
        batch_size = self.training_settings["batch_size"]
        n_batches = dataset_size // batch_size

        loss_fun = NestedMWPMLoss.apply
        n_epochs = self.training_settings["tot_epochs"]

        # initialise simulations and graph settings
        m_nearest_nodes = self.graph_settings["m_nearest_nodes"]
        n_node_features = self.graph_settings["n_node_features"]
        power = self.graph_settings["power"]

        sims = self.initialise_simulations()

        # generate validation syndromes
        n_val_graphs = self.training_settings["validation_set_size"]
        val_syndromes, val_flips, n_val_identities = self.create_test_set(
            n_graphs=n_val_graphs,
        )
        
        # train model
        for epoch in range(current_epoch, n_epochs):
            print(f"Epoch {epoch}")
            self.model.train()
            
            # set potential seed
            if self.simul_seed:
                seed = self.simul_seed + epoch + int(1e8)
            else:
                seed = None
            train_loss = 0
            epoch_n_graphs = 0
            epoch_n_trivial = 0
            
            
            # if epoch > 0:
            #     for i, (name, p) in enumerate(self.model.named_parameters()):
            #         print(f"Parameter tensor {name}:")
            #         print(p.grad)
            #         print(f"Mean of parameter tensor {i}:")
            #         print(torch.mean(p.grad))
            for _ in range(n_batches):
                
                # simulate data as we go
                sim = random.choice(sims)
                syndromes, flips, n_trivial = sim.generate_syndromes(
                    use_for_mwpm=True, seed=seed
                )
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
                edge_index, edge_weights, edge_classes = self.model(
                    x,
                    edge_index,
                    edge_attr,
                    detector_labels,
                    batch_labels,
                )
                loss = loss_fun(
                    *edge_index,
                    *edge_weights,
                    *edge_classes,
                    flips,
                )
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * n_graphs
                epoch_n_graphs += n_graphs

            # compute losses and logical accuracy
            # ------------------------------------

            # train
            train_loss /= epoch_n_graphs

            # validation
            val_accuracy = self.evaluate_test_set(
                val_syndromes,
                val_flips,
                n_val_identities,
                n_graphs=n_val_graphs,
            )

            # save training attributes after every epoch
            self.epoch = epoch
            self.training_history["epoch"] = epoch
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_accuracy"].append(val_accuracy)

            if self.save_model:
                self.save_model_w_training_settings()
            
            print(f"Loss: {train_loss:.6f}")
            print(f"Accuracy: {val_accuracy:.2f}")

    def get_training_metrics(self):

        train_loss = self.training_history["train_loss"]
        val_accuracy = self.training_history["val_accuracy"]

        return train_loss, val_accuracy

    def check_performance(self, model=None, n_graphs=1e4):

        if model is None:
            model = self.model

        # create a test set
        syndromes, flips, n_identities = self.create_test_set(
            error_rate=1e-3,
            n_graphs=n_graphs,
        )
        
        # print statistics about dataset 
        syndrome_shapes = [s.shape[0] for s in syndromes]
        flips_per_syndrome = [f.sum() for f in flips]       
        n_syndromes = sum(syndrome_shapes)
        n_logical_flips = sum(flips_per_syndrome)

        print(
            f"We have {n_syndromes} non-trivial syndromes in a set of {int(n_graphs)} samples."
        )
        print(
            f"{int(n_logical_flips)} of the syndromes correspond to a logical operator, {int(n_logical_flips / n_syndromes * 100)}%."
        )

        # run inference
        preds = []
        for syndrome in syndromes:

            x, edge_index, edge_attr, batch_labels, detector_labels = get_batch_of_graphs(
                syndrome, self.graph_settings["m_nearest_nodes"], device=self.device
            )
            edge_index, edge_weights, edge_classes = model(
                x,
                edge_index,
                edge_attr,
                detector_labels,
                batch_labels,
            )

            # preds.append(predict_mwpm(edge_index, edge_weights, edge_classes, batch_labels))
            preds.append(predict_mwpm_nested(edge_index, edge_weights, edge_classes))
        
        preds = np.concatenate(preds)
        flips = np.concatenate(flips)
        n_correct = (preds == flips).sum()

        # accuracy and logical accuracy
        accuracy = n_correct / n_syndromes
        logical_accuracy = (n_correct + n_identities) / n_graphs

        # confusion plot
        true_identity = ((preds == 0) & (flips == 0)).sum()
        true_flip = ((preds == 1) & (flips == 1)).sum()
        false_identity = ((preds == 0) & (flips == 1)).sum()
        false_flip = ((preds == 1) & (flips == 0)).sum()

        confusion_data = [[true_identity, false_identity], [false_flip, true_flip]]
        df_confusion = pd.DataFrame(
            confusion_data,
            index=["Predicted 0", "Predicted 1"],
            columns=["True 0", "True 1"],
        )
        
        # print results
        print(
            f"We have an accuracy of {int(accuracy * 100)}% and a logical accuracy of {int(logical_accuracy * 100)}%."
        )

        return accuracy, logical_accuracy, df_confusion