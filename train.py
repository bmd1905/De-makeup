import yaml
from yaml.loader import SafeLoader

from model.trainer import Trainer
from loader.dataloader import Dataloader

if __name__ == '__main__':
    # Open the file and load the file
    with open('config.yml') as f:
        config = yaml.load(f, Loader=SafeLoader)
    
    # Configuration model
    trainer = Trainer(config)

    # Load data
    dataloader = Dataloader(config)
    train_ds, val_ds, test_ds = dataloader.loader()

    # Train
    trainer.train(train_ds, val_ds)

