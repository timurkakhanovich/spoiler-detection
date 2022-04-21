import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# Text CNN ---------------------------------------
class KernelBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1_2 = nn.Conv1d(in_channels, out_channels, 
                                 kernel_size, padding='same')
        self.bn1_2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2_2 = nn.Conv1d(in_channels, out_channels, 
                                 kernel_size, padding='same')
        self.bn2_2 = nn.BatchNorm1d(num_features=out_channels)
        
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x):
        out1 = self.pool(F.relu(self.bn1_2(self.conv1_2(x))))
        out2 = self.pool(F.relu(self.bn2_2(self.conv2_2(x))))

        return torch.cat([out1, out2], dim=1)
    

class TextCNN(nn.Module):
    def __init__(self, tokenizer, hid_size=64):
        super().__init__()
        self.vocab_size = tokenizer.vocab_size
        self.emb_dim = tokenizer.dim
        self.hid_size = hid_size
        
        self.embeddings = nn.Embedding(num_embeddings=self.vocab_size,
                                       embedding_dim=self.emb_dim, padding_idx=10000)
        self.embeddings.weight.data.copy_(torch.from_numpy(tokenizer.vectors))
        
        self.emb_dropout = nn.Dropout(p=0.1)
        
        self.ker_block2 = KernelBlock(self.emb_dim, hid_size, kernel_size=2)
        self.ker_block3 = KernelBlock(self.emb_dim, hid_size, kernel_size=3)
        self.ker_block4 = KernelBlock(self.emb_dim, hid_size, kernel_size=4)
        self.ker_block5 = KernelBlock(self.emb_dim, hid_size, kernel_size=5)
        
        self.final_conv = nn.Conv1d(in_channels=8*hid_size, out_channels=2*hid_size,
                                    kernel_size=3, padding='same')
        self.bn = nn.BatchNorm1d(num_features=2*hid_size)

        self.dropout = nn.Dropout(p=0.2)

        self.final_fc = nn.Linear(in_features=2*hid_size, out_features=1)
    
    def forward(self, x):
        embed = self.emb_dropout(self.embeddings(x)).permute(0, 2, 1)

        out2 = self.ker_block2(embed)
        out3 = self.ker_block3(embed)
        out4 = self.ker_block4(embed)
        out5 = self.ker_block5(embed)
        
        cat_out = self.dropout(torch.cat([out2, out3, out4, out5], dim=1))

        conv_out = F.relu(self.bn(self.final_conv(cat_out))).squeeze(dim=2)
        result_out = self.final_fc(conv_out)

        return torch.sigmoid(result_out)
# -------------------------------------------


# TCN ---------------------------------------
class DilatedSubBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = (kernel_size-1)*dilation

        self.conv = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            padding=self.padding, dilation=dilation
        ))
        self.dropout = nn.Dropout(p=0.2)

    def clip(self, x):
        return x[:, :, :-self.padding].contiguous()

    def forward(self, x):
        conv_out = F.relu(self.clip(self.conv(x)))

        return self.dropout(conv_out)


class DilatedCausalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=[1, 2, 4]):
        super().__init__()
        layers = [
            DilatedSubBlock(in_channels, out_channels, kernel_size, dilations[0]), 
            DilatedSubBlock(out_channels, out_channels, kernel_size, dilations[1]), 
            DilatedSubBlock(out_channels, out_channels, kernel_size, dilations[2]), 
        ]

        self.conv_layers = nn.Sequential(*layers)
        self.skip_connection = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return F.relu(self.skip_connection(x) + self.conv_layers(x))


class TCN(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.vocab_size = tokenizer.vocab_size
        self.emb_dim = tokenizer.dim
        
        self.emb_layer = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.emb_dim, padding_idx=10000)
        self.emb_layer.weight.data.copy_(torch.from_numpy(tokenizer.vectors))
        
        self.emb_dropout = nn.Dropout(p=0.2)

        self.tcn_block1 = DilatedCausalConv(
            in_channels=self.emb_dim, out_channels=128, kernel_size=3, dilations=[1, 2, 4]
        )
        self.tcn_block2 = DilatedCausalConv(
            in_channels=128, out_channels=64, kernel_size=3, dilations=[1, 2, 4]
        )

        self.global_max_pool = nn.AdaptiveMaxPool1d(output_size=1)

        self.fc1 = nn.Linear(in_features=64, out_features=1)
    
    def forward(self, x):
        embedding_out = self.emb_dropout(
            self.emb_layer(x).permute(0, 2, 1)
        )

        conv_out1 = self.tcn_block1(embedding_out)
        conv_out2 = self.tcn_block2(conv_out1)

        conv_encoded = self.global_max_pool(conv_out2).squeeze(2)
        
        fc_out1 = self.fc1(conv_encoded)
        
        return torch.sigmoid(fc_out1)
# ------------------------------------------------

# LSTM v1---------------------------------------
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
