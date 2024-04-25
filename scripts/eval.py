from pathlib import Path
import torch
import argparse
import sys
sys.path.append("../")
from src.models import GraphNN, SimpleGraphNNV4, SimpleGraphNNV6
from src.evaluation import ModelEval
from src.utils import plot_syndrome
import os
import logging
logging.disable(sys.maxsize)
os.environ["QECSIM_CFG"] = "/cephyr/users/fridafj/Alvis"
#import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

def main():
    # command line parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configuration", required=True)
    #parser.add_argument("-s", "--save", required=False, action="store_true")
    args = parser.parse_args() 
    
    # create a model
    #model = GraphNN()
    #model = SimpleGraphNNV4()
    model = SimpleGraphNNV6()
    #model = GATNN()
    config = Path(args.configuration)
    
    # check if model should be saved
    
    # train model
    evaluator = ModelEval(model, config=config)
    remove_virtual = True
    #trainer.train_warmup()
    syndromes, flips, n_id = evaluator.create_split_test_set()
    if remove_virtual:
        syndromes, flips = remove_virtual_nodes(syndromes, flips)
    wrong_syndromes, wrong_flips = evaluator.evaluate_test_set(syndromes, flips, n_id)
    #wrong_ratio, all_ratio, wrong_virtual_ratio = calculate_ratios(syndromes, wrong_syndromes, "z")
    #num_z, num_x, num_z_wrong, num_x_wrong = get_node_counts(syndromes, wrong_syndromes)
    #save_ratios(num_z, num_x, num_z_wrong, num_x_wrong)
    

def remove_virtual_nodes(syndromes, flips):
    label = {"z": 3, "x": 1}
    for i in range(len(syndromes)):
        syndrome = syndromes[i].astype(np.float32)
        flip = flips[i]
        _even_odd_all = np.count_nonzero(syndrome == label["z"], axis=(1, 2, 3)) & 1
        all_even = np.logical_not(_even_odd_all)
        no_virtual = syndrome[all_even,...]
        no_virtual_flips = flip[all_even]
        #print(no_virtual.shape)
        syndromes[i] = no_virtual
        flips[i] = no_virtual_flips
    return syndromes, flips
    
def calculate_ratios(syndromes, wrong_syndromes, experiment):
    label = {"z": 3, "x": 1}
    even_odd_all = 0
    n_syndromes = 0
    for syndrome in syndromes:
        syndrome = syndrome.astype(np.float32)
        _even_odd_all = np.count_nonzero(syndrome == label[experiment], axis=(1, 2, 3)) & 1
        even_odd_all += _even_odd_all.sum()
        _n_syndromes = syndrome.shape[0]
        n_syndromes += _n_syndromes
    even_odd_wrong = np.count_nonzero(wrong_syndromes == label[experiment], axis=(1, 2, 3)) & 1
    all_syndromes_ratio = even_odd_all/n_syndromes
    wrong_syndromes_ratio = even_odd_wrong.sum()/wrong_syndromes.shape[0]
    wrong_virtual = even_odd_wrong.sum()
    wrong_virtual_ratio = wrong_virtual/even_odd_all
    print("wrong: ", wrong_syndromes_ratio)
    print("all: ", all_syndromes_ratio)
    print("wrong virtual ratio: ", wrong_virtual_ratio)
    return wrong_syndromes_ratio, all_syndromes_ratio, wrong_virtual_ratio


def save_syndromes(wrong_syndromes, wrong_flips):
    file_syndrome = datetime.now().strftime("%y%m%d-%H%M%S") + "_syndrome.npy"
    file_flip = datetime.now().strftime("%y%m%d-%H%M%S") + "_flip.npy"
    with open(file_syndrome, 'wb') as f:
        np.save(f, wrong_syndromes)
    with open(file_flip, 'wb') as f:
        np.save(f, wrong_flips)

def save_ratios(n_z, n_x, wrong_z, wrong_x):
    dict = {"total_z": n_z,
            "total_x": n_x,
            "wrong_z": wrong_z,
            "wrong_x": wrong_x}
    file_dict = datetime.now().strftime("%y%m%d-%H%M%S") + "_dict.npy"
    with open(file_dict, 'wb') as f:
        np.save(f, dict)


def get_node_counts(syndromes, wrong_syndromes):
    label = {"z": 3, "x": 1}
    n_syndromes = 0
    num_nodes_z = None
    num_nodes_x = None
    for syndrome in syndromes:
        syndrome = syndrome.astype(np.float32)
        _num_nodes_z = np.count_nonzero(syndrome == label["z"], axis=(1, 2, 3))
        _num_nodes_x = np.count_nonzero(syndrome == label["x"], axis=(1, 2, 3))
        if num_nodes_z is not None:
            num_nodes_z = np.concatenate((num_nodes_z, _num_nodes_z), axis=None)
            num_nodes_x = np.concatenate((num_nodes_x, _num_nodes_x), axis=None)
        else:
            num_nodes_z = _num_nodes_z
            num_nodes_x = _num_nodes_x
    num_z_wrong = np.count_nonzero(wrong_syndromes == label["z"], axis=(1, 2, 3))
    num_x_wrong = np.count_nonzero(wrong_syndromes == label["x"], axis=(1, 2, 3))
    return num_nodes_z, num_nodes_x, num_z_wrong, num_x_wrong


if __name__ == "__main__":
    main()    