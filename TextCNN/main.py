from pathlib import Path
import gc
from functools import partial

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import wandb

from TextCNN.modules.data.dataset import (
    load_goodreads_dataset,
    load_IMDB_dataset,
    load_tv_tropes_dataset,
    pad_collate
)

from TextCNN.modules.pipeline_config.utils import (
    Metrics,
    set_seed, 
    set_split,
    get_class_weights,
    BPEMB_EN,
    PAD_IDX
)

from TextCNN.modules.pipeline_config.model_config import ModelConfig
from TextCNN.modules.pipeline_config.train import train


def main_command_line_args():
    set_seed(42)
    CHECKPOINT_PATH = Path('CNN_checkpoints/')
    
    configs = ModelConfig(CHECKPOINT_PATH).get_arguments()
    
    dataset_name = 'IMDB & Goodreads'
    goodreads_df = load_goodreads_dataset(
        '../Datasets/Goodreads_dataset/goodreads_dataset.json'
    )
    IMDB_df = load_IMDB_dataset(
        '../Datasets/IMDB_dataset/IMDB_reviews.json'
    )
    reviews_df = pd.concat([goodreads_df, IMDB_df])
    
    device = configs['device']
    model = configs['model']
    next(iter(model.parameters())).requires_grad = False
    
    optimizer = configs['optimizer']
    scheduler = configs['scheduler']
    num_epochs = configs['num_epochs']
    start_epoch = configs['start_epoch']
    history = configs['history']
    lr = configs['lr']
    batch_size = configs['batch_size']
    
    class_weights = get_class_weights(reviews_df, device)
    
    collator = partial(pad_collate, pad_value=PAD_IDX)
    data = set_split(
        reviews_df, BPEMB_EN, test_size=0.2, 
        batch_size=batch_size, train_shuffle=True, collator=collator
    )
    del reviews_df
    gc.collect()
    
    metrics = Metrics()
    criterion = nn.BCELoss(reduction='none')
    
    config = {
        'learning_rate': lr,
        'batch_size': batch_size,
        'embedding_dim': BPEMB_EN.dim,
        'hid_size': model.hid_size
    }
    
    with wandb.init(project='Spoiler-classification', entity='timkakhanovich', config=config) as run:
        wandb.watch(model, log='all', log_freq=150)
        
        
        model_artifact = wandb.Artifact(
            'TextCNN-v1', type='model'
        )
        model_filename = 'model.pt'
        torch.save(model.state_dict(), model_filename)
        model_artifact.add_file(model_filename)

        wandb.save(model_filename)
        
        
        train(
            model, data.dataloaders, criterion, class_weights, optimizer, metrics, lr, 
            scheduler=scheduler, num_epochs=num_epochs, start_epoch=start_epoch, prev_metrics=history, 
            device=device, folder_for_checkpoints=CHECKPOINT_PATH, dataset_name=dataset_name, run=run
        )


if __name__ == '__main__':
    main_command_line_args()
