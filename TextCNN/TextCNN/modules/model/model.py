import torch
import torch.nn as nn
import torch.nn.functional as F


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
