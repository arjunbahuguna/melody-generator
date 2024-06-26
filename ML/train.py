import sys
import os

# Add the ML directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
from ML.preprocessor import MelodyPreprocessor
from ML.transformer import MelodyTransformer, generate_square_subsequent_mask, create_padding_mask
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import os
import pickle

def train_model(midi_folder_path, batch_size=4, num_epochs=10, learning_rate=0.0001, accumulation_steps=8, model_dir="ML/models"):
    preprocessor = MelodyPreprocessor(midi_folder_path, batch_size=batch_size)
    dataloader = preprocessor.create_training_dataset()
    
    total_samples = len(dataloader.dataset)
    num_steps_per_epoch = len(dataloader)
    print(f"Total number of samples: {total_samples}")
    print(f"Number of steps per epoch: {num_steps_per_epoch}")

    num_tokens = preprocessor.number_of_tokens_with_padding
    dim_model = 256
    num_heads = 4
    num_encoder_layers = 4
    num_decoder_layers = 4
    dropout = 0.1
    
    model = MelodyTransformer(
        num_tokens=num_tokens, 
        dim_model=dim_model, 
        num_heads=num_heads, 
        num_encoder_layers=num_encoder_layers, 
        num_decoder_layers=num_decoder_layers, 
        dropout=dropout
    ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, batch in enumerate(dataloader):
            src = batch["input"].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            tgt = batch["target"].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Ensure src and tgt_input have the same batch size and correct values
            assert src.shape[0] == tgt_input.shape[0], "Batch sizes of src and tgt_input do not match"
            
            # Check if any index in src or tgt_input exceeds the vocabulary size
            if torch.any(src >= num_tokens) or torch.any(tgt_input >= num_tokens):
                continue  # Skip this batch

            src_mask = generate_square_subsequent_mask(src.size(1)).to(src.device)
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(tgt_input.device)
            src_padding_mask = create_padding_mask(src).to(src.device)
            tgt_padding_mask = create_padding_mask(tgt_input).to(tgt_input.device)
            
            optimizer.zero_grad()
            
            with autocast():
                output = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, 
                               src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
                loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            print(f'Epoch {epoch+1}, Step {i+1}, Loss: {loss.item()}')
        
        print(f'Epoch {epoch+1}, Average Loss: {epoch_loss / len(dataloader)}')

        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, f"melody_transformer_model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved as '{model_path}'")

        tokenizer_path = os.path.join(model_dir, "tokenizer.pkl")
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(preprocessor.tokenizer, f)
        print(f"Tokenizer saved as '{tokenizer_path}'")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a Melody Transformer model.')
    parser.add_argument('--midi_folder_path', type=str, required=True, help='Path to the folder containing MIDI files.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer.')
    parser.add_argument('--accumulation_steps', type=int, default=8, help='Number of gradient accumulation steps.')
    parser.add_argument('--model_dir', type=str, default="ML/models", help='Directory to save the trained model.')
    
    args = parser.parse_args()
    
    train_model(
        midi_folder_path=args.midi_folder_path, 
        batch_size=args.batch_size, 
        num_epochs=args.num_epochs, 
        learning_rate=args.learning_rate,
        accumulation_steps=args.accumulation_steps,
        model_dir=args.model_dir
    )

# Example Usage
#python3 ML/train.py --midi_folder_path ML/data --batch_size 4 --num_epochs 20 --learning_rate 0.0001 --accumulation_steps 4 --model_dir ML/models