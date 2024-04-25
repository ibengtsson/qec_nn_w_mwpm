from pathlib import Path
import torch
import argparse
import sys
sys.path.append("../")
from src.models import GraphNN, SimpleGraphNNV4, GATNN
from src.training import LSTrainer, LSTrainer_v2
import os
import logging
logging.disable(sys.maxsize)
os.environ["QECSIM_CFG"] = "/cephyr/users/fridafj/Alvis"
import numpy as np
import torch.nn as nn
import torch_geometric.nn as nng
import os
from pathlib import Path
from datetime import datetime
import random
from typing import Callable

from src.utils import parse_yaml, inference, get_misclassified_syndromes
from src.simulations import SurfaceCodeSim
from src.graph import get_batch_of_graphs
from src.local_search import LocalSearch

class ModelEval:
    def __init__(
        self,
        model: nn.Module,
        config: os.PathLike = None
        ):
        paths, graph_settings, eval_settings = parse_yaml(config)
        self.saved_model_path = paths["saved_model_path"]
        self.graph_settings = graph_settings
        self.training_settings = eval_settings
        if "cuda" in eval_settings["device"]:
            self.device = torch.device(
                eval_settings["device"] if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device("cpu")
        
        if not (
            self.device == torch.device("cuda") or self.device == torch.device("cpu")
        ):
            torch.cuda.set_device(self.device)
        # move model to correct device
        self.model = model.to(self.device)
        self.load_trained_model()


    def load_trained_model(self):
        model_path = Path(self.saved_model_path)
        saved_attributes = torch.load(model_path, map_location=self.device)

        # update attributes and load model with trained weights
        self.training_history_prev = saved_attributes["training_history"]
        # older models do not have the attribute "best_val_accuracy"
        # if not "best_val_score" in self.training_history:
        #     self.training_history["best_val_score"] = -1
        self.model.load_state_dict(saved_attributes["model"])

        # only keep best found weights
        self.optimal_weights = saved_attributes["model"]

    # create a split set of syndromes used for training, one error prob only
    def create_split_test_set(self):
        
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

        return syndromes, flips, n_id
    

    def evaluate_test_set(self, syndromes, flips, n_identities, n_removed):
        n_graphs = self.training_settings["dataset_size"]
        n_graphs = n_graphs-n_removed
        # is n_graphs=n_syndromes? i dont think so
        m_nearest_nodes = self.graph_settings["m_nearest_nodes"]
        n_correct_preds = 0
        wrong_syndromes = None
        wrong_flips = []
        for syndrome, flip in zip(syndromes, flips):
            # this should maybe be changed to ls_inference and graphs generated before
            _n_correct_preds, _ = inference(
                self.model,
                syndrome,
                flip,
                experiment=self.graph_settings["experiment"],
                m_nearest_nodes=m_nearest_nodes,
                device=self.device,
            )
            n_correct_preds += _n_correct_preds
            _wrong_syndromes, _wrong_flips = get_misclassified_syndromes(
                self.model,
                syndrome,
                flip,
                experiment=self.graph_settings["experiment"],
                m_nearest_nodes=m_nearest_nodes,
                device=self.device,
            )
            if wrong_syndromes is not None:
                if _wrong_syndromes.shape[0] > 0:
                    wrong_syndromes = np.concatenate((wrong_syndromes, _wrong_syndromes), axis=0)
                    wrong_flips = np.concatenate((wrong_flips, _wrong_flips), axis=0)
            else:
                if _wrong_syndromes.shape[0] > 0:
                    wrong_syndromes = _wrong_syndromes
                    wrong_flips = _wrong_flips
            # if _wrong_syndromes.shape[0] > 0:
            #     wrong_syndromes.append(_wrong_syndromes)
            #     wrong_flips.append(_wrong_flips)

        # compute logical failure rate
        failure_rate = (n_graphs - n_correct_preds - n_identities) / n_graphs
        print(f"We have a logical failure rate of {failure_rate}.")
        return wrong_syndromes, wrong_flips