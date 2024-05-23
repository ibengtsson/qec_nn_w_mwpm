from pathlib import Path
import torch
import argparse
import sys
sys.path.append("../")
from src.models import GraphNN, SimpleGraphNNV4, SimpleGraphNNV6, GraphAttention, GraphAttentionV2, GraphAttentionV3
from src.evaluation import ModelEval
#from src.utils import plot_syndrome
import os
import logging
logging.disable(sys.maxsize)
os.environ["QECSIM_CFG"] = "/cephyr/users/fridafj/Alvis"
#import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
plt.style.use("science")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def main():
    # command line parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configuration", required=True)
    #parser.add_argument("-s", "--save", required=False, action="store_true")
    args = parser.parse_args() 
    
    # create a model
    #model = GraphNN()
    model = SimpleGraphNNV4()
    #model = SimpleGraphNNV6()
    #model = GraphAttentionV3()
    #model = GATNN()
    config = Path(args.configuration)
    
    # check if model should be saved
    
    # train model
    evaluator = ModelEval(model, config=config)
    #trainer.train_warmup()
    # tot_id = 0
    # for i in range(15):
    syndromes, flips, n_id = evaluator.create_split_test_set()
    #     tot_id += n_id
    #     print(n_id)
    
    # avg_id = tot_id/15
    # print(avg_id)

    n_removed = 0
    wrong_syndromes, wrong_flips, preds, flip_arr  = evaluator.evaluate_test_set(syndromes, flips, n_id, n_removed)
    n_correct = (preds == flip_arr).sum()
    n_graphs = len(flip_arr)
    acc = n_correct/n_graphs
    TP = np.sum(np.logical_and(preds == 1, flip_arr == 1))
    TN = np.sum(np.logical_and(preds == 0, flip_arr == 0))
    FP = np.sum(np.logical_and(preds == 1, flip_arr == 0))
    FN = np.sum(np.logical_and(preds == 0, flip_arr == 1))
    sens = TP/(TP+FN)
    spec = TN/(TN+FP)
    #bal_acc = (sens+spec)/2
    #print(bal_acc)
    #cm = confusion_matrix(flip_arr, preds, normalize='true')
    #print(cm)
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm,
    #                          display_labels=["I", "X"])
    #print(disp.text_kw)
    #save_folder = Path("..\..\Figures\Plots")
    # colors = sns.color_palette("crest", as_cmap=True)
    # text_width = 5.9066
    # column_width = 0.45 * text_width
    # aspect_ratio = 1
    # #aspect_ratio = 16/9
    # figsize_tw = (text_width, 1/aspect_ratio * text_width)
    # figsize_cw = (column_width, 1/aspect_ratio * column_width)
    # fig, ax = plt.subplots(figsize=figsize_cw)
    # disp = sns.heatmap(cm*100, annot=True, fmt = '.1f', 
    #                    square=1, ax=ax, cbar=False, cmap="crest",xticklabels=["I","X"],yticklabels=["I","X"])
    # for t in disp.texts: t.set_text(t.get_text() + "\%")
    # disp.set_xlabel('Predicted label')
    # disp.set_ylabel('True label')
    # plt.tick_params(
    # axis='both',          # changes apply to the x-axis
    # which='both',      # both major and minor ticks are affected
    # bottom=False,      # ticks along the bottom edge are off
    # top=False,
    # right=False,
    # left=False,         # ticks along the top edge are off
    # labelbottom=True,
    # labelleft=True)
    #disp.plot(cmap=colors,colorbar=False, ax=ax, values_format= '.0%')
    #fig.tight_layout()
    #plt.savefig("conf_bal.pdf", format="pdf")
    #plt.show()
    #num_wrong_flip = np.count_nonzero(wrong_flips == 1)
    #num_wrong_no_flip = np.count_nonzero(wrong_flips == 0)
    #print(num_wrong_flip)
    #print(num_wrong_no_flip)
    #syndromes, flips, n_removed = remove_virtual_nodes(syndromes, flips)
    #wrong_syndromes, wrong_flips = evaluator.evaluate_test_set(syndromes, flips, n_id, n_removed)
    #wrong_ratio, all_ratio, wrong_virtual_ratio = calculate_ratios(syndromes, wrong_syndromes, "z")
    #num_z, num_x, num_z_wrong, num_x_wrong = get_node_counts(syndromes, wrong_syndromes)
    #save_ratios(num_z, num_x, num_z_wrong, num_x_wrong)
    

def remove_virtual_nodes(syndromes, flips):
    label = {"z": 3, "x": 1}
    n_removed = 0
    for i in range(len(syndromes)):
        syndrome = syndromes[i].astype(np.float32)
        flip = flips[i]
        _even_odd_all = np.count_nonzero(syndrome == label["z"], axis=(1, 2, 3)) & 1
        n_removed += _even_odd_all.sum()
        all_even = np.logical_not(_even_odd_all)
        no_virtual = syndrome[all_even,...]
        no_virtual_flips = flip[all_even]
        #print(no_virtual.shape)
        syndromes[i] = no_virtual
        flips[i] = no_virtual_flips
    return syndromes, flips, n_removed
    
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