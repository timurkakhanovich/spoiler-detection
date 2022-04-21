import torch
import torch.nn as nn
import torch.nn.functional as F


# LSTM v1 ---------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, tokenizer, hid_size=64):
        super().__init__()
        self.vocab_size = tokenizer.vocab_size
        self.emb_dim = tokenizer.dim
        self.emb_dropout = nn.Dropout(p=0.2)
        self.hid_size = hid_size
        
        self.emb_layer = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.emb_dim)
        self.emb_layer.weight.data.copy_(torch.from_numpy(tokenizer.vectors))
        
        self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hid_size, 
                            num_layers=2, batch_first=True, dropout=0.2)
        
        self.fc1 = nn.Linear(in_features=2*self.hid_size, out_features=1)
    
    def forward(self, x):
        embedding_out = self.emb_dropout(self.emb_layer(x))
        
        _, (hiddens, c) = self.lstm(embedding_out)
        hid = torch.cat([hiddens[0, :, :], hiddens[1, :, :]], dim=1)
        
        fc_out = self.fc1(hid)

        return torch.sigmoid(fc_out)
# ------------------------------------------------


# BiLSTM ---------------------------------------
class BiLSTMModel(nn.Module):
    def __init__(self, tokenizer, hid_size=64):
        super().__init__()
        self.vocab_size = tokenizer.vocab_size
        self.emb_dim = tokenizer.dim
        self.hid_size = hid_size
        
        self.emb_layer = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.emb_dim)
        self.emb_layer.weight.data.copy_(torch.from_numpy(tokenizer.vectors))
        self.emb_dropout = nn.Dropout(p=0.1)
        
        self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hid_size, 
                            num_layers=2, batch_first=True, 
                            bidirectional=True, dropout=0.2)
        
        self.global_max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        
        self.fc1 = nn.Linear(in_features=self.hid_size*4, out_features=1)
    
    def forward(self, x):
        embedding_out = self.emb_dropout(self.emb_layer(x))
        lstm_out, _ = self.lstm(embedding_out)
        lstm_out = lstm_out.permute(0, 2, 1)
        lstm_encoded = torch.cat((self.global_max_pool(lstm_out), 
                                  self.global_avg_pool(lstm_out)), dim=1).squeeze(2)
        fc_out1 = self.fc1(lstm_encoded)

        return torch.sigmoid(fc_out1)
# ------------------------------------------------
