import os
import json
import mido
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

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

        # Tokenize and encode the parsed melodies
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
        self.tokenizer.fit(melody)
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

                # Debug: print sequences and their lengths
                print(f"Input sequence: {input_seq}, length: {len(input_seq)}")
                print(f"Padded input sequence: {padded_input_seq}, length: {len(padded_input_seq)}")
                print(f"Target sequence: {target_seq}, length: {len(target_seq)}")
                print(f"Padded target sequence: {padded_target_seq}, length: {len(padded_target_seq)}")

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

if __name__ == "__main__":
    # Usage example
    preprocessor = MelodyPreprocessor("/home/arjbah/Projects/melody-generator/data", batch_size=1)
    training_dataloader = preprocessor.create_training_dataset()
    
    # Print the first batch to verify
    for batch in training_dataloader:
        print(batch["input"], batch["target"])
        break