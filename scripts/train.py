from pathlib import Path
import torch
import argparse
import sys
sys.path.append("../")
from src.models import GraphNN, MWPMLoss
from src.training import ModelTrainer

def main():
    
    # command line parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configuration", required=True)
    args = parser.parse_args() 
    
    # create a model
    model = GraphNN()
    loss_fun = MWPMLoss.apply
    config = Path(args.configuration)
    
    trainer = ModelTrainer(model, loss_fun, config=config)
    trainer.train(warmup=True)
    trainer.train()
    
    loss, logical_accuracy = trainer.get_training_metrics()
    
    print(loss)
    print(logical_accuracy)
    

if __name__ == "__main__":
    main()