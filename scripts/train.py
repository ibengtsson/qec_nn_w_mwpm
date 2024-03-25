from pathlib import Path
import torch
import argparse
import sys
sys.path.append("../")
from src.models import GraphNN, SimpleGraphNN
from src.training import LSTrainer, LSTrainer_v2
import os
import logging
logging.disable(sys.maxsize)
os.environ["QECSIM_CFG"] = "/cephyr/users/fridafj/Alvis"


def main():
    # command line parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configuration", required=True)
    parser.add_argument("-s", "--save", required=False, action="store_true")
    args = parser.parse_args() 
    
    # create a model
    model = GraphNN()
    #model = SimpleGraphNN()
    config = Path(args.configuration)
    
    # check if model should be saved
    if args.save:
        save_or_not = args.save
        print("Model will be saved after each epoch.")
    else:
        save_or_not = False
        print("Model will not be saved.")
    
    # train model
    trainer = LSTrainer_v2(model, config=config, save_model=save_or_not)
    #trainer.train_warmup()
    trainer.train()
    
    train_score, val_score, time = trainer.get_training_metrics()
    
    print(train_score)
    print(val_score)
    print(time)
    

if __name__ == "__main__":
    main()    

# def main():
    
#     # command line parsing
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-c", "--configuration", required=True)
#     parser.add_argument("-s", "--save", required=False, action="store_true")
#     args = parser.parse_args() 
    
#     # create a model
#     model = GraphNN()
#     loss_fun = MWPMLoss_v2.apply
#     config = Path(args.configuration)
    
#     # check if model should be saved
#     if args.save:
#         save_or_not = args.save
#         print("Model will be saved after each epoch.")
#     else:
#         save_or_not = False
#         print("Model will not be saved.")
    
#     # train model
#     trainer = ModelTrainer(model, loss_fun, config=config, save_model=save_or_not)
#     trainer.train(warmup=True)
#     trainer.train()
    
#     loss, logical_accuracy = trainer.get_training_metrics()
    
#     print(loss)
#     print(logical_accuracy)