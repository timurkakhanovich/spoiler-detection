import json
import os
from tqdm import tqdm

import pandas as pd
from pandas import json_normalize

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SpoilerDataset(Dataset):
    def __init__(self, tokenizer, data, labels):
        super().__init__()

        self.tokenizer = tokenizer
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        encoded_sent = torch.tensor(self.tokenizer.encode_ids(self.data[idx]))

        return encoded_sent, self.labels[idx]
    
    def __len__(self):
        return len(self.data)


def pad_collate(batch, pad_value):
    batch_tokens_lengths = torch.tensor([x[0].shape[0] for x in batch])
    max_len_per_batch = torch.max(batch_tokens_lengths)
    
    # Truncate.  
    max_len_per_batch = max_len_per_batch if max_len_per_batch <= 512 else 512
    
    # Extend to even max_len.  
    additive = (max_len_per_batch % 2)
    max_len_per_batch += additive
    lengths_to_pad = max_len_per_batch - batch_tokens_lengths - additive
    
    X_batch = torch.stack([
        F.pad(x[0], pad=(0, val_to_pad), value=pad_value) 
        for x, val_to_pad in zip(batch, lengths_to_pad)
    ]).type(torch.int64)
    
    y_batch = torch.tensor([x[1] for x in batch]).type(torch.int64)

    return X_batch, y_batch


def load_goodreads_dataset(data_path):
    with open(data_path, 'r') as fin:
        data = json.load(fin)
    
    sents = data['0'] + data['1']
    spoilers = [0] * len(data['0']) + [1] * len(data['1'])
    
    return pd.DataFrame(data=zip(sents, spoilers), 
                        columns=('review_text', 'is_spoiler'))


def load_IMDB_dataset(data_path):
    with open(data_path, 'r') as fin:
        reviews = [json.loads(data) for data in fin.readlines()]
        reviews_df = json_normalize(reviews)
        del reviews

    return reviews_df[['review_text', 'is_spoiler']]


def load_tv_tropes_dataset(data_path):
    data_files = ['train.balanced.csv', 'test.balanced.csv', 
                  'dev1.balanced.csv', 'dev2.balanced.csv']
    
    dfs = list()

    for data_file in data_files:
        dfs.append(pd.read_csv(os.path.join(data_path, data_file)))
    
    return pd.concat(dfs, ignore_index=True).rename(
        columns={'sentence': 'review_text', 'spoiler': 'is_spoiler'}
    )[['review_text', 'is_spoiler']]
