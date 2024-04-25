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
    #trainer.train_warmup()
    syndromes, flips, n_id = evaluator.create_split_test_set()
    
    wrong_syndromes, wrong_flips = evaluator.evaluate_test_set(syndromes, flips, n_id)
    label = {"z": 3, "x": 1}
    experiment = "z"
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
    print("wrong: ", wrong_syndromes_ratio)
    print("all: ", all_syndromes_ratio)

    # file_syndrome = datetime.now().strftime("%y%m%d-%H%M%S") + "_syndrome.npy"
    # file_flip = datetime.now().strftime("%y%m%d-%H%M%S") + "_flip.npy"
    # with open(file_syndrome, 'wb') as f:
    #     np.save(f, wrong_syndromes)
    # with open(file_flip, 'wb') as f:
    #     np.save(f, wrong_flips)
    


    

if __name__ == "__main__":
    main()    