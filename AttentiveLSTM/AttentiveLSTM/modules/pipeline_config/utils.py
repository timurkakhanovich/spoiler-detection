from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from AttentiveLSTM.modules.data.dataset import SpoilerDataset, pad_collate

from bpemb import BPEmb


# Tokenizer.  
BPEMB_EN = BPEmb(lang='en', dim=300, vs=50000, add_pad_emb=True)
BPEMB_EN.vocab_size = len(BPEMB_EN.emb.index_to_key)
BPEMB_EN.vs = len(BPEMB_EN.emb.index_to_key)
PAD_IDX = BPEMB_EN.vs - 1


@dataclass
class DataCollections:
    datasets: Dict[str, SpoilerDataset]
    dataloaders: Dict[str, torch.utils.data.dataloader.DataLoader]
    

@dataclass
class Validation:
    loss: float
    scores: Dict[str, float]
    

class Metrics:
    def __init__(self):
        super().__init__()
    
    def __call__(self, y_pred, y_true):
        y_pred = y_pred.detach().cpu()
        y_true = y_true.detach().cpu()

        return {
            'precision': precision_score(y_true, y_pred > 0.5),
            'recall': recall_score(y_true, y_pred > 0.5),
            'f1-score': f1_score(y_true, y_pred > 0.5),
            'roc_auc_score': roc_auc_score(y_true, y_pred)
        }


def set_seed(seed=42):
    '''
    Set seed for reproducibility.
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def set_device(in_device):
    if in_device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

        if in_device == 'cuda':
            print('CUDA is not available. Switched to CPU')
    
    return device

    
def get_class_weights(reviews_df, device):
    true_class_count = reviews_df['is_spoiler'].sum()
    false_class_count = len(reviews_df) - true_class_count
    
    w_0 = len(reviews_df) / (2 * false_class_count)
    w_1 = len(reviews_df) / (2 * true_class_count)
    
    return torch.FloatTensor([w_0, w_1]).to(device)
    

def set_split(reviews_df, tokenizer, test_size=0.2, 
              batch_size=8, train_shuffle=False, collator=None):
    X = reviews_df['review_text'].to_numpy()
    y = reviews_df['is_spoiler'].to_numpy().astype(np.int32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, shuffle=train_shuffle, random_state=42
    )

    datasets = {
        'train': SpoilerDataset(tokenizer, X_train, y_train),
        'val': SpoilerDataset(tokenizer, X_val, y_val)
    }

    dataloaders = {
        'train': DataLoader(datasets['train'], shuffle=True, batch_size=batch_size, 
                            num_workers=4, collate_fn=collator), 
        'val': DataLoader(datasets['val'], shuffle=False, batch_size=128, 
                          num_workers=4, collate_fn=collator), 
    }
    
    return DataCollections(datasets, dataloaders)


def validate_model(model, val_dataloader, criterion, class_weights, metrics, 
                    device=torch.device('cpu'), return_train=False):
    model.eval()
    running_loss = 0.0

    predictions = torch.FloatTensor([]).to(device)
    targets = torch.FloatTensor([]).to(device)
    with torch.inference_mode():
        print('\n')
        for batch_idx, sample in enumerate(val_dataloader):
            if batch_idx % 100 == 0 or batch_idx == len(val_dataloader) - 1:
                print(f'==> Batch: {batch_idx}/{len(val_dataloader)}')
            
            X = sample['text'].to(device)
            y_true = sample['label'].to(device, dtype=torch.float32)
            attn_mask = sample['attention_mask'].to(device)
            
            y_pred = model(X, attn_mask).squeeze()
            
            # Weighted BCELoss.  
            target_weights = class_weights[y_true.data.view(-1).long()].view_as(y_true)
            loss = criterion(y_pred, y_true)
            loss *= target_weights
            loss = torch.mean(loss)

            running_loss += loss.item()
            predictions = torch.cat((predictions, y_pred))
            targets = torch.cat((targets, y_true))

        all_metrics_score = metrics(predictions, targets)
        running_loss /= len(val_dataloader)
        
    if return_train:
        model.train()

    return Validation(running_loss, all_metrics_score)
