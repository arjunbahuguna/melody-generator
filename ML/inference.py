import sys
import os

# Add the ML directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from ML.transformer import MelodyTransformer, generate_square_subsequent_mask, create_padding_mask
import pickle
import mido

def load_model_and_tokenizer(model_path, tokenizer_path, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout=0.1):
    model = MelodyTransformer(
        num_tokens=num_tokens, 
        dim_model=dim_model, 
        num_heads=num_heads, 
        num_encoder_layers=num_encoder_layers, 
        num_decoder_layers=num_decoder_layers, 
        dropout=dropout
    ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()

    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    return model, tokenizer

def generate_melody(model, tokenizer, start_sequence, max_length=100):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_sequence = torch.tensor(tokenizer.transform(start_sequence), dtype=torch.long).unsqueeze(0).to(device)

    generated_sequence = input_sequence

    for _ in range(max_length):
        src_mask = generate_square_subsequent_mask(generated_sequence.size(1)).to(device)
        src_padding_mask = create_padding_mask(generated_sequence).to(device)
        
        with torch.no_grad():
            output = model(generated_sequence, generated_sequence, src_mask=src_mask, tgt_mask=src_mask,
                           src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=src_padding_mask)
        
        next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(0)
        generated_sequence = torch.cat((generated_sequence, next_token), dim=1)

        if next_token.item() == 0:
            break

    generated_melody = tokenizer.inverse_transform(generated_sequence.squeeze().cpu().numpy())

    return generated_melody

def save_melody_to_midi(melody, output_path):
    import mido
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    for note in melody:
        note, time = map(int, note.split(','))
        track.append(mido.Message('note_on', note=note, velocity=64, time=time))

    mid.save(output_path)
    print(f"Melody saved as '{output_path}'")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate melody using trained Transformer model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer file.')
    parser.add_argument('--start_sequence', type=str, nargs='+', required=True, help='Starting sequence for the melody generation.')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length of the generated melody.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the generated melody as a MIDI file.')

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        num_tokens=871,  # This should match the number of tokens used during training
        dim_model=256,  # This should match the model's dimension used during training
        num_heads=4,  # This should match the number of heads used during training
        num_encoder_layers=4,  # This should match the number of encoder layers used during training
        num_decoder_layers=4,  # This should match the number of decoder layers used during training
        dropout=0.1  # This should match the dropout rate used during training
    )

    generated_melody = generate_melody(
        model=model,
        tokenizer=tokenizer,
        start_sequence=args.start_sequence,
        max_length=args.max_length
    )

    save_melody_to_midi(generated_melody, args.output_path)

# Example Usage
#python3 ML/inference.py --model_path ML/models/melody_transformer_model_epoch_1.pth --tokenizer_path ML/models/tokenizer.pkl --start_sequence "60,0" "62,0" "64,0" --max_length 100 --output_path generated_melody.mid