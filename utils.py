import os
import librosa
import torch
from bidict import bidict
from torch.utils.data import Dataset

DATA_BASE_PATH = "dataset/an4"
LABEL_DIR = "txt"
DATA_DIR = "wav"

VOCABULARY = bidict({" ": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6,
                           "g": 7, "h": 8, "i": 9, "j": 10, "k": 11, "l": 12, "m": 13,
                           "n": 14, "o": 15, "p": 16, "q": 17, "r": 18, "s": 19, "t": 20,
                           "u": 21, "v": 22, "w": 23, "x": 24, "y": 25, "z": 26,
                           "": 27})

LAS_VOCABULARY = bidict({" ": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6,
                           "g": 7, "h": 8, "i": 9, "j": 10, "k": 11, "l": 12, "m": 13,
                           "n": 14, "o": 15, "p": 16, "q": 17, "r": 18, "s": 19, "t": 20,
                           "u": 21, "v": 22, "w": 23, "x": 24, "y": 25, "z": 26,
                           "": 27, "<sos>":28})


def load_data(data_type: str = "train"):
    data = {}
    data_path = os.path.join(DATA_BASE_PATH, data_type)
    i = 0
    for file in os.listdir(os.path.join(data_path, DATA_DIR)):
        if not file.endswith(".wav"):
            continue
        # extract label
        label_path = os.path.join(data_path, LABEL_DIR, os.path.splitext(
                file)[0] + ".txt")
        with open(label_path, "r") as label:
            data[i] = {'label': label.read().lower(), 'audio_path': os.path.join(data_path,
                                                                        DATA_DIR, file)}
        i += 1

    return data


class AN4(Dataset):
    def __init__(self, data_dict=None, audio_paths=None):
        self.data_dict = data_dict
        self.audio_paths = audio_paths

    def __len__(self):
        if self.audio_paths:
            return len(self.audio_paths)
        return len(self.data_dict)

    def __getitem__(self, idx):
        if self.audio_paths:
            item = self.audio_paths[idx]
            wav, sr = librosa.load(item)
            return wav, len(wav)

        item = self.data_dict[idx]
        audio_path = item["audio_path"]
        transcription = item["label"]
        transcription_length = len(transcription)

        # load audio
        wav, sr = librosa.load(audio_path)

        # Uncomment to enable data augmentation
        # noise_factor = 0.003
        # # data augmentation adding noise
        # data = data + noise_factor * np.random.normal(0, 1, len(data))
        #
        # # shifting audio
        # shift = np.random.randint(0, 100)
        # data = np.roll(data, shift)
        #
        # # changing pitch
        # pitch_change = np.random.randint(-5, 5)
        # data = librosa.effects.pitch_shift(data, sr=sr, n_steps=pitch_change)

        return wav, len(wav), transcription, transcription_length


def collate_pad(batch):
    # Collate function that pads audio samples in a batch
    batch_size = len(batch)
    audio_lengths = torch.tensor([sample[1] for sample in batch])
    transcription_lengths = torch.tensor([sample[3] for sample in batch])
    max_audio_length = max(audio_lengths)
    max_transcription_length = max(transcription_lengths)

    padded_audio = torch.zeros((batch_size, max_audio_length))
    padded_transcript = torch.full((batch_size, max_transcription_length), VOCABULARY[""])
    for i, (audio, length, transcriptions, _) in enumerate(batch):
        padded_audio[i, :length] = torch.Tensor(audio)

        for j, char_ in enumerate(transcriptions):
            padded_transcript[i, j] = VOCABULARY[char_]

    return padded_audio, audio_lengths, padded_transcript, transcription_lengths