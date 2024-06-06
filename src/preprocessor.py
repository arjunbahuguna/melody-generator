import json
import mido
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class MelodyPreprocessor:
    """
    a class for preprocessing melodies for a transformer model.

    this class takes melodies from a midi file, tokenizes and encodes them, 
    and prepares pytorch datasets for training sequence-to-sequence models.
    """

    def __init__(self, midi_file_path, batch_size=32):
        """
        initializes the melody preprocessor.

        parameters:
            midi_file_path (str): path to the midi file.
            batch_size (int): size of each batch in the dataset.
        """
        self.midi_file_path = midi_file_path
        self.batch_size = batch_size
        self.tokenizer = LabelEncoder()
        self.max_melody_length = None
        self.number_of_tokens = None

    @property
    def number_of_tokens_with_padding(self):
        """
        returns the number of tokens in the vocabulary including padding.

        returns:
            int: the number of tokens in the vocabulary including padding.
        """
        return self.number_of_tokens + 1

    def create_training_dataset(self):
        """
        preprocesses the melody dataset and creates sequence-to-sequence
        training data.

        returns:
            dataloader: a pytorch dataloader containing input-target
                pairs suitable for training a sequence-to-sequence model.
        """
        # load and parse the midi file
        melody = self._load_midi()
        parsed_melody = self._parse_melody(melody)
        
        # tokenize and encode the parsed melody
        tokenized_melody = self._tokenize_and_encode_melody(parsed_melody)
        
        # set the maximum melody length and number of tokens
        self._set_max_melody_length([tokenized_melody])
        self._set_number_of_tokens()
        
        # create input and target sequence pairs
        input_sequences, target_sequences = self._create_sequence_pairs([tokenized_melody])
        
        # create a dataset and dataloader
        dataset = MelodyDataset(input_sequences, target_sequences)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def _load_midi(self):
        """
        loads the melody data from a midi file.

        returns:
            list: a list of note and time tuples from the midi file.
        """
        midi = mido.MidiFile(self.midi_file_path)
        melody = []
        for msg in midi:
            # filter out meta messages and note_off messages
            if not msg.is_meta and msg.type == 'note_on' and msg.velocity > 0:
                note = msg.note
                time = msg.time
                melody.append((note, time))
        return melody

    def _parse_melody(self, melody):
        """
        parses a list of note and time tuples into a list of strings.

        parameters:
            melody (list): a list of note and time tuples.

        returns:
            list: a list of string representations of notes.
        """
        parsed_melody = []
        for note, time in melody:
            parsed_melody.append(f"{note},{time}")
        return parsed_melody

    def _tokenize_and_encode_melody(self, melody):
        """
        tokenizes and encodes a list of melodies.

        parameters:
            melody (list): a list of melodies to be tokenized and encoded.

        returns:
            tokenized_melody: a list of tokenized and encoded melodies.
        """
        self.tokenizer.fit(melody)
        tokenized_melody = self.tokenizer.transform(melody)
        return tokenized_melody

    def _set_max_melody_length(self, melodies):
        """
        sets the maximum melody length based on the dataset.

        parameters:
            melodies (list): a list of tokenized melodies.
        """
        self.max_melody_length = max([len(melody) for melody in melodies])

    def _set_number_of_tokens(self):
        """
        sets the number of tokens based on the tokenizer.
        """
        self.number_of_tokens = len(self.tokenizer.classes_)

    def _create_sequence_pairs(self, melodies):
        """
        creates input-target pairs from tokenized melodies.

        parameters:
            melodies (list): a list of tokenized melodies.

        returns:
            tuple: two numpy arrays representing input sequences and target sequences.
        """
        input_sequences, target_sequences = [], []
        for melody in melodies:
            for i in range(1, len(melody)):
                input_seq = melody[:i]
                target_seq = melody[1 : i + 1]  # shifted by one time step
                padded_input_seq = self._pad_sequence(input_seq)
                padded_target_seq = self._pad_sequence(target_seq)
                input_sequences.append(padded_input_seq)
                target_sequences.append(padded_target_seq)
        return np.array(input_sequences), np.array(target_sequences)

    def _pad_sequence(self, sequence):
        """
        pads a sequence to the maximum sequence length.

        parameters:
            sequence (list): the sequence to be padded.

        returns:
            list: the padded sequence.
        """
        return sequence + [0] * (self.max_melody_length - len(sequence))

class MelodyDataset(Dataset):

    def __init__(self, input_sequences, target_sequences):
        """
        parameters:
            input_sequences (numpy.ndarray): input sequences for the model.
            target_sequences (numpy.ndarray): target sequences for the model.
        """
        self.input_sequences = input_sequences
        self.target_sequences = target_sequences

    def __len__(self):
        """
        returns:
            int: number of samples in the dataset.
        """
        return len(self.input_sequences)

    def __getitem__(self, idx):
        """
        parameters:
            idx (int): index of the sample to retrieve.

        returns:
            dict: a dictionary containing the input and target sequences.
        """
        sample = {
            "input": torch.tensor(self.input_sequences[idx], dtype=torch.long),
            "target": torch.tensor(self.target_sequences[idx], dtype=torch.long),
        }
        return sample

if __name__ == "__main__":
    # usage example
    preprocessor = MelodyPreprocessor("/path/to/midi_file", batch_size=32)
    training_dataloader = preprocessor.create_training_dataset()
    
    # print the first batch to verify
    for batch in training_dataloader:
        print(batch["input"], batch["target"])
        break
