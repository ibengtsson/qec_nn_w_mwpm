from pathlib import Path
import argparse
import sys
sys.path.append("../")
from src.simple_training import SimpleTrainer
import os
if "alvis" in os.uname().nodename:
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
    
    # initialise model trainer
    trainer = SimpleTrainer(config=config, save_model=save_or_not, seeds=False)
    
    # measure a benchmark
    # print("BEFORE TRAINING:")
    # _, _, confusion_df = trainer.check_performance()
    # print(confusion_df)

    # train model
    trainer.train()
    
    # check performance
    # print("AFTER TRAINING:")
    # _, _, confusion_df = trainer.check_performance()
    
    # print(confusion_df)
    

if __name__ == "__main__":
    main()

