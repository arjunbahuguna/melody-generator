import torch
import torch.nn as nn
import torch.optim as optim

class MelodyTransformer(nn.Module):
    def __init__(self, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout=0.1):
        super(MelodyTransformer, self).__init__()
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.positional_encoding = PositionalEncoding(dim_model, dropout)
        self.transformer = nn.Transformer(
            d_model=dim_model, 
            nhead=num_heads, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dropout=dropout
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
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

if __name__ == "__main__":
    import math
    from melody_preprocessor import MelodyPreprocessor
    
    # hyperparameters
    num_tokens = 128  # match the vocabulary size from the tokenizer
    dim_model = 512
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dropout = 0.1
    batch_size = 32
    num_epochs = 10
    
    # initialize the preprocessor
    preprocessor = MelodyPreprocessor("path/to/midi_file", batch_size=batch_size)
    dataloader = preprocessor.create_training_dataset()
    
    # initialize the model, loss function, and optimizer
    model = MelodyTransformer(
        num_tokens=preprocessor.number_of_tokens_with_padding, 
        dim_model=dim_model, 
        num_heads=num_heads, 
        num_encoder_layers=num_encoder_layers, 
        num_decoder_layers=num_decoder_layers, 
        dropout=dropout
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignoring padding index
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # training loop
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            src = batch["input"]
            tgt = batch["target"]
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            src_mask = generate_square_subsequent_mask(src.size(1))
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1))
            
            optimizer.zero_grad()
            
            output = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
            
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
            loss.backward()
            
            optimizer.step()
        
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
    # save the trained model
    torch.save(model.state_dict(), "melody_transformer_model.pth")
