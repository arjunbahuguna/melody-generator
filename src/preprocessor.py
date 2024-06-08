import os
import mido
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class MelodyPreprocessor:
    def __init__(self, midi_folder_path, batch_size=32):
        self.midi_folder_path = midi_folder_path
        self.batch_size = batch_size
        self.tokenizer = LabelEncoder()
        self.max_melody_length = None
        self.number_of_tokens = None

    @property
    def number_of_tokens_with_padding(self):
        return self.number_of_tokens + 1

    def create_training_dataset(self):
        melodies = self._load_midi_files()
        parsed_melodies = [self._parse_melody(melody) for melody in melodies]
        all_tokens = [token for melody in parsed_melodies for token in melody]
        self.tokenizer.fit(all_tokens)
        tokenized_melodies = [self._tokenize_and_encode_melody(parsed_melody) for parsed_melody in parsed_melodies]
        self._set_max_melody_length(tokenized_melodies)
        self._set_number_of_tokens()
        input_sequences, target_sequences = self._create_sequence_pairs(tokenized_melodies)
        dataset = MelodyDataset(input_sequences, target_sequences)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def _load_midi_files(self):
        melodies = []
        for file_name in os.listdir(self.midi_folder_path):
            if file_name.endswith('.mid'):
                midi = mido.MidiFile(os.path.join(self.midi_folder_path, file_name))
                melody = []
                for msg in midi:
                    if not msg.is_meta and msg.type == 'note_on' and msg.velocity > 0:
                        note = msg.note
                        time = msg.time
                        melody.append((note, time))
                melodies.append(melody)
        return melodies

    def _parse_melody(self, melody):
        parsed_melody = []
        for note, time in melody:
            parsed_melody.append(f"{note},{time}")
        return parsed_melody

    def _tokenize_and_encode_melody(self, melody):
        tokenized_melody = self.tokenizer.transform(melody)
        return tokenized_melody

    def _set_max_melody_length(self, melodies):
        self.max_melody_length = max([len(melody) for melody in melodies])

    def _set_number_of_tokens(self):
        self.number_of_tokens = len(self.tokenizer.classes_)

    def _create_sequence_pairs(self, melodies):
        input_sequences, target_sequences = [], []
        for melody in melodies:
            for i in range(1, len(melody)):
                input_seq = melody[:i]
                target_seq = melody[1:i + 1]
                padded_input_seq = self._pad_sequence(input_seq)
                padded_target_seq = self._pad_sequence(target_seq)
                input_sequences.append(padded_input_seq)
                target_sequences.append(padded_target_seq)
        return np.array(input_sequences), np.array(target_sequences)

    def _pad_sequence(self, sequence):
        if isinstance(sequence, np.ndarray):
            sequence = sequence.tolist()
        return np.pad(sequence, (0, self.max_melody_length - len(sequence)), mode='constant')

class MelodyDataset(Dataset):
    def __init__(self, input_sequences, target_sequences):
        self.input_sequences = input_sequences
        self.target_sequences = target_sequences

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        sample = {
            "input": torch.tensor(self.input_sequences[idx], dtype=torch.long),
            "target": torch.tensor(self.target_sequences[idx], dtype=torch.long),
        }
        return sample

if __name__ == "__main__":
    preprocessor = MelodyPreprocessor("/path/to/midi/files", batch_size=1)
    training_dataloader = preprocessor.create_training_dataset()
    
    for batch in training_dataloader:
        print(batch["input"], batch["target"])
        break
