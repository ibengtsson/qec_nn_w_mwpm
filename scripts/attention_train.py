from pathlib import Path
import argparse
import sys
sys.path.append("../")
from src.attention_training import ModelTrainer
import os
if "alvis" in os.uname().nodename:
    os.environ["QECSIM_CFG"] = "/cephyr/users/isakbe/Alvis"
    SEED = False
else:
    SEED = True

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
    
    # initialise model trainer
    trainer = ModelTrainer(config=config, save_model=save_or_not, seeds=SEED)

    # train model
    trainer.train()

if __name__ == "__main__":
    main()

