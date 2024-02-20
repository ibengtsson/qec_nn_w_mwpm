from pathlib import Path
import torch
import sys
sys.path.append("../")
from src.models import GraphNN, MWPMLoss
from src.training import ModelTrainer

def main():
    
    torch.manual_seed(111)
    model = GraphNN()
    loss_fun = MWPMLoss.apply
    config = Path('../configs/default_config.yaml')
    config = None
    
    trainer = ModelTrainer(model, loss_fun, config=config)
    trainer.train(warmup=True)
    trainer.train()
    
    loss, logical_accuracy = trainer.get_training_metrics()
    
    print(loss)
    print(logical_accuracy)
    

if __name__ == "__main__":
    main()