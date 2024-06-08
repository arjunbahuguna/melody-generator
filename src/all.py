import os
import json
import mido
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import math
import torch.nn as nn
import torch.optim as optim

class MelodyPreprocessor:
    """
    A class for preprocessing melodies for a transformer model.

    This class takes melodies from MIDI files, tokenizes and encodes them,
    and prepares PyTorch datasets for training sequence-to-sequence models.
    """

    def __init__(self, midi_folder_path, batch_size=32):
        """
        Initializes the melody preprocessor.

        Parameters:
            midi_folder_path (str): Path to the folder containing MIDI files.
            batch_size (int): Size of each batch in the dataset.
        """
        self.midi_folder_path = midi_folder_path
        self.batch_size = batch_size
        self.tokenizer = LabelEncoder()
        self.max_melody_length = None
        self.number_of_tokens = None

    @property
    def number_of_tokens_with_padding(self):
        """
        Returns the number of tokens in the vocabulary including padding.

        Returns:
            int: The number of tokens in the vocabulary including padding.
        """
        return self.number_of_tokens + 1

    def create_training_dataset(self):
        """
        Preprocesses the melody dataset and creates sequence-to-sequence
        training data.

        Returns:
            DataLoader: A PyTorch DataLoader containing input-target
                pairs suitable for training a sequence-to-sequence model.
        """
        # Load and parse MIDI files
        melodies = self._load_midi_files()
        parsed_melodies = [self._parse_melody(melody) for melody in melodies]

        # Fit tokenizer on the entire dataset
        all_melodies = [item for sublist in parsed_melodies for item in sublist]
        self.tokenizer.fit(all_melodies)
        tokenized_melodies = [self._tokenize_and_encode_melody(parsed_melody) for parsed_melody in parsed_melodies]

        # Set the maximum melody length and number of tokens
        self._set_max_melody_length(tokenized_melodies)
        self._set_number_of_tokens()

        # Debug: print the max melody length
        print(f"Max melody length: {self.max_melody_length}")

        # Create input and target sequence pairs
        input_sequences, target_sequences = self._create_sequence_pairs(tokenized_melodies)

        # Create a dataset and DataLoader
        dataset = MelodyDataset(input_sequences, target_sequences)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def _load_midi_files(self):
        """
        Loads melody data from MIDI files in the specified folder.

        Returns:
            list: A list of lists containing note and time tuples from MIDI files.
        """
        melodies = []
        for file_name in os.listdir(self.midi_folder_path):
            if file_name.endswith('.mid'):
                midi = mido.MidiFile(os.path.join(self.midi_folder_path, file_name))
                melody = []
                for msg in midi:
                    # Filter out meta messages and note_off messages
                    if not msg.is_meta and msg.type == 'note_on' and msg.velocity > 0:
                        note = msg.note
                        time = msg.time
                        melody.append((note, time))
                melodies.append(melody)
        return melodies

    def _parse_melody(self, melody):
        """
        Parses a list of note and time tuples into a list of strings.

        Parameters:
            melody (list): A list of note and time tuples.

        Returns:
            list: A list of string representations of notes.
        """
        parsed_melody = []
        for note, time in melody:
            parsed_melody.append(f"{note},{time}")
        return parsed_melody

    def _tokenize_and_encode_melody(self, melody):
        """
        Tokenizes and encodes a list of melodies.

        Parameters:
            melody (list): A list of melodies to be tokenized and encoded.

        Returns:
            tokenized_melody: A list of tokenized and encoded melodies.
        """
        tokenized_melody = self.tokenizer.transform(melody)
        return tokenized_melody

    def _set_max_melody_length(self, melodies):
        """
        Sets the maximum melody length based on the dataset.

        Parameters:
            melodies (list): A list of tokenized melodies.
        """
        self.max_melody_length = max([len(melody) for melody in melodies])

    def _set_number_of_tokens(self):
        """
        Sets the number of tokens based on the tokenizer.
        """
        self.number_of_tokens = len(self.tokenizer.classes_)

    def _create_sequence_pairs(self, melodies):
        """
        Creates input-target pairs from tokenized melodies.

        Parameters:
            melodies (list): A list of tokenized melodies.

        Returns:
            tuple: Two numpy arrays representing input sequences and target sequences.
        """
        input_sequences, target_sequences = [], []
        for melody in melodies:
            for i in range(1, len(melody)):
                input_seq = melody[:i]
                target_seq = melody[1:i + 1]  # Shifted by one time step
                padded_input_seq = self._pad_sequence(input_seq)
                padded_target_seq = self._pad_sequence(target_seq)
                input_sequences.append(padded_input_seq)
                target_sequences.append(padded_target_seq)

        return np.array(input_sequences), np.array(target_sequences)

    def _pad_sequence(self, sequence):
        """
        Pads a sequence to the maximum sequence length.

        Parameters:
            sequence (list or numpy.ndarray): The sequence to be padded.

        Returns:
            numpy.ndarray: The padded sequence.
        """
        if isinstance(sequence, np.ndarray):
            sequence = sequence.tolist()
        return np.pad(sequence, (0, self.max_melody_length - len(sequence)), mode='constant')

class MelodyDataset(Dataset):
    def __init__(self, input_sequences, target_sequences):
        """
        Parameters:
            input_sequences (numpy.ndarray): Input sequences for the model.
            target_sequences (numpy.ndarray): Target sequences for the model.
        """
        self.input_sequences = input_sequences
        self.target_sequences = target_sequences

    def __len__(self):
        """
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.input_sequences)

    def __getitem__(self, idx):
        """
        Parameters:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the input and target sequences.
        """
        sample = {
            "input": torch.tensor(self.input_sequences[idx], dtype=torch.long),
            "target": torch.tensor(self.target_sequences[idx], dtype=torch.long),
        }
        return sample

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

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

if __name__ == "__main__":
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
    preprocessor = MelodyPreprocessor("/home/arjbah/Projects/melody-generator/data", batch_size=batch_size)
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
