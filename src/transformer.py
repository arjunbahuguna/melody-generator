import math
import torch
import torch.nn as nn

class MelodyTransformer(nn.Module):
    def __init__(self, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout=0.1):
        super(MelodyTransformer, self).__init__()
        self.embedding = nn.Embedding(num_tokens, dim_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(dim_model, dropout)
        self.transformer = nn.Transformer(
            d_model=dim_model, 
            nhead=num_heads, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(dim_model, num_tokens)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src_embedded = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src_embedded = self.positional_encoding(src_embedded)
        tgt_embedded = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        tgt_embedded = self.positional_encoding(tgt_embedded)
        
        transformer_out = self.transformer(
            src_embedded, tgt_embedded, 
            src_mask=src_mask, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask
        )
        out = self.fc_out(transformer_out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
        pe = torch.zeros(1, max_len, dim_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_padding_mask(seq, pad_idx=0):
    return (seq == pad_idx)
