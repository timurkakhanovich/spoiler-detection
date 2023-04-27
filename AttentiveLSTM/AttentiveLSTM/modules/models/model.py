import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, lstm_out, attn_mask):
        '''
        Attention forward.  
        :param lstm_out: LSTM outputs, 
        :param final_state: final state of LSTM layer
        :param attn_mask: 3-dimensional tensor of masked features (batch, seq_len)
        :returns: concated hidden state and attention feature
        '''
        
        # Get valid final states of sequences.  
        final_state_ids = torch.sum(attn_mask, dim=1) - 1
        final_states = lstm_out[range(lstm_out.size(0)), final_state_ids]
        
        # Attention mask for tensor to select valid vectors.  
        attn_mask = attn_mask.unsqueeze(-1).repeat(1, 1, lstm_out.size(-1))
        
        # Seq vectors.  
        valid_features = (attn_mask * lstm_out)  # (batch, seq_len, hid_size)  
        
        a_t = torch.bmm(
            valid_features, final_states.unsqueeze(2)
        ).squeeze()  # (batch, seq_len)  
        
        # Replace 'null' with '-inf' values for softmax.  
        a_t[a_t == 0] = -float('inf')
        p_t = F.softmax(a_t, dim=1)  # (batch, seq_len)  
        
        weighted_features = torch.bmm(
            p_t.unsqueeze(1), lstm_out
        ).squeeze()  # (batch, hid_size)  
        
        return torch.cat((final_states, weighted_features), dim=1)


class AttentiveBiLSTM(nn.Module):
    def __init__(self, tokenizer, hid_size=64):
        super().__init__()
        
        self.vocab_size = tokenizer.vocab_size
        self.emb_dim = tokenizer.dim
        self.hid_size = hid_size
        
        self.emb_layer = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.emb_dim
        )
        self.emb_layer.weight.data.copy_(torch.from_numpy(tokenizer.vectors))
        
        self.lstm = nn.LSTM(
            input_size=self.emb_dim, hidden_size=self.hid_size, 
            num_layers=2, batch_first=True, 
            bidirectional=True, dropout=0.2
        )
        
        self.attention = AttentionLayer()
        self.fc = nn.Linear(in_features=self.hid_size*4, out_features=1)
    
    def forward(self, sample, attn_mask):
        emb_out = self.emb_layer(sample)
        lstm_out, _ = self.lstm(emb_out)
        
        attn_out = self.attention(lstm_out, attn_mask)
        fc_out = self.fc(attn_out)
        
        return torch.sigmoid(fc_out)
