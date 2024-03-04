from pathlib import Path
import argparse
import sys
sys.path.append("../")
from src.training import ModelTrainer
import os
os.environ["QECSIM_CFG"] = "/cephyr/users/isakbe/Alvis"

def main():
    
    # command line parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configuration", required=True)
    parser.add_argument("-s", "--save", required=False, action="store_true")
    args = parser.parse_args() 
    
    # get a config
    config = Path(args.configuration)
    
    # check if model should be saved
    if args.save:
        save_or_not = args.save
        print("Model will be saved after each epoch.")
    else:
        save_or_not = False
        print("Model will not be saved.")
    
    # train model
    trainer = ModelTrainer(config=config, save_model=save_or_not)
    trainer.train(warmup=True)
    trainer.train()
    
    loss, logical_accuracy = trainer.get_training_metrics()
    
    print(loss)
    print(logical_accuracy)

if __name__ == "__main__":
    main()

