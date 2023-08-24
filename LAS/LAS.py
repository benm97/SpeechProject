from itertools import groupby

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from Jasper.utils import AN4
from jiwer import wer, cer
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from nemo.collections.asr.modules import SpectrogramAugmentation
from scipy.ndimage import shift
from torch.utils.data import DataLoader

from utils import LAS_VOCABULARY

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)

    def forward(self, inputs):
        outputs, _ = self.lstm(inputs)
        outputs = self.batch_norm(outputs.transpose(1, 2)).transpose(1, 2)
        outputs = self.relu(outputs)
        outputs, _ = self.lstm2(outputs)
        outputs = self.batch_norm2(outputs.transpose(1, 2)).transpose(1, 2)
        outputs = self.relu(outputs)
        return outputs


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.W_a = nn.Linear(hidden_size, hidden_size)
        self.V_a = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        energy = torch.tanh(self.W_a(encoder_outputs) + decoder_hidden.unsqueeze(1))
        attention_scores = self.V_a(energy).squeeze(2)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        return context_vector, attention_weights


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, inputs, decoder_hidden):
        decoder_hidden, decoder_cell = self.lstm(inputs, decoder_hidden)
        decoder_hidden = self.batch_norm(decoder_hidden)
        decoder_hidden = self.relu(decoder_hidden)
        output = self.fc(decoder_hidden)
        return output, (decoder_hidden, decoder_cell)


class LAS(pl.LightningModule):
    def __init__(self, num_mels, hidden_size, num_classes, spec_augment=False):
        super(LAS, self).__init__()
        self.encoder = Encoder(num_mels, hidden_size)
        self.attention = Attention(hidden_size)
        self.decoder = Decoder(hidden_size + num_classes, hidden_size, num_classes)
        self.num_classes = num_classes
        self.preprocessor = AudioToMelSpectrogramPreprocessor(features=num_mels)
        self.spec_augmentation = SpectrogramAugmentation()
        self.spec_augment = spec_augment
        # self.loss = CTCLoss(num_classes - 1, zero_infinity=True, reduction="mean_batch")
        self.loss = nn.CrossEntropyLoss(ignore_index=LAS_VOCABULARY[""])
        self.lr = 0.001

    def forward(self, input_signal, input_signal_length, n_steps, prev_correct_char=None):

        processed_signal, processed_signal_length = self.preprocessor(input_signal=input_signal,
                                                                      length=input_signal_length)

        if self.spec_augment and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        processed_signal = processed_signal.transpose(1, 2)
        encoder_outputs = self.encoder(processed_signal)
        decoder_hidden = torch.zeros(input_signal.size(0),
                                     self.decoder.hidden_size).to(device)
        decoder_cell = torch.zeros(input_signal.size(0),
                                   self.decoder.hidden_size).to(device)
        outputs = []

        if prev_correct_char is None:  # during inference
            # prev_token = torch.Tensor(np.full((encoder_inputs.shape[0], ), LAS_VOCAB[
            # "<sos>"]))
            prev_token = torch.Tensor(np.zeros((input_signal.shape[0],
                                                self.num_classes))).to(device)
            prev_token[:, LAS_VOCABULARY["<sos>"]] = 1

            for t in range(n_steps):
                context_vector, _ = self.attention(encoder_outputs, decoder_hidden)
                context_vector = context_vector.reshape(context_vector.shape[0],
                                                        context_vector.shape[2])
                decoder_input = torch.cat((prev_token, context_vector),
                                          dim=1)
                output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input, (
                    decoder_hidden, decoder_cell))
                outputs.append(output)
                prev_token = output

        else:
            for i in range(n_steps):
                context_vector, attention_weights = self.attention(encoder_outputs,
                                                                   decoder_hidden)
                context_vector = context_vector.reshape(context_vector.shape[0],
                                                        context_vector.shape[2])
                prev_token = torch.Tensor(np.zeros((input_signal.size(0),
                                                    self.num_classes))).to(device)
                for sample_ind, token_ind in enumerate(prev_correct_char[:, i]):
                    prev_token[int(sample_ind), int(token_ind)] = 1
                decoder_input = torch.cat((prev_token, context_vector), dim=1).float()
                output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input,
                                                                      (decoder_hidden,
                                                                       decoder_cell))
                outputs.append(output)

        outputs = torch.stack(outputs, dim=1).to(device)
        predictions = outputs.argmax(dim=2, keepdim=False)

        return outputs, processed_signal_length, predictions

    def transcribe(self, audio_paths, batch_size=4):
        self.eval()
        test_dataset = AN4(audio_paths=audio_paths)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=self.__collate_pad_audio_only, )
        hypothesis = []
        for input_signal, input_lengths in test_dataloader:
            log_probs, encoded_len, predictions = self.forward(input_signal=input_signal,
                                                               input_signal_length=input_lengths)
            hypothesis.extend(self.__decode(predictions, encoded_len))
        return hypothesis

    def __decode(self, predictions, prediction_lengths):
        hypothesis = []
        for pred, length in zip(predictions, prediction_lengths):
            all_tokens = [LAS_VOCABULARY.inverse[int(token)] for token in pred[:length]]
            all_tokens_remove_duplicate = [key for key, _ in groupby(all_tokens)]
            sentence = "".join(all_tokens_remove_duplicate)
            hypothesis.append(sentence)
        return hypothesis

    def __collate_pad_audio_only(self, batch):
        # Collate function that pads audio samples in a batch
        batch_size = len(batch)
        audio_lengths = torch.tensor([sample[1] for sample in batch])
        max_audio_length = max(audio_lengths)

        padded_audio = torch.zeros((batch_size, max_audio_length))
        for i, (audio, length) in enumerate(batch):
            padded_audio[i, :length] = torch.Tensor(audio)

        return padded_audio.to(device), audio_lengths.to(device)

    def training_step(self, batch, batch_idx):
        signal, signal_len, transcript, transcript_len = batch
        transcript_shifted = torch.tensor(shift(transcript.cpu(), cval=LAS_VOCABULARY["<sos>"], shift=[0, 1]))
        log_probs, encoded_len, predictions = self.forward(input_signal=signal, input_signal_length=signal_len,
                                                           prev_correct_char=transcript_shifted,
                                                           n_steps=max(transcript_len))
        # loss_value = self.loss(
        #     log_probs=log_probs, targets=transcript, input_lengths=transcript_len, target_lengths=transcript_len
        # )
        loss_value = self.loss(log_probs.view(-1, self.num_classes), transcript.view(-1))
        self.log('learning_rate', self.optimizer.param_groups[0]['lr'])
        self.log('train_loss', loss_value)
        return loss_value

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = {
            'scheduler': lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200),
            'interval': 'epoch',  # Adjust the learning rate at the end of each epoch
            'frequency': 1  # Adjust the learning rate every epoch
        }
        return [self.optimizer], [self.scheduler]

    def validation_step(self, batch, batch_idx):
        signal, signal_len, transcript, transcript_len = batch
        log_probs, encoded_len, predictions = self.forward(input_signal=signal, input_signal_length=signal_len,
                                                           n_steps=max(transcript_len))
        loss_value = self.loss(log_probs.view(-1, self.num_classes), transcript.view(-1))
        # Greedy Decoding
        hypothesis = self.__decode(predictions, encoded_len)
        ground_truth = ["".join([LAS_VOCABULARY.inverse[int(token)] for token in sentence]) for sentence in transcript]

        wer_ = wer(ground_truth, hypothesis)
        cer_ = cer(ground_truth, hypothesis)
        # self._wer.update(
        #     predictions=log_probs, targets=transcript, target_lengths=transcript_len, predictions_lengths=encoded_len
        # )
        # wer, wer_num, wer_denom = self._wer.compute()
        # self._wer.reset()
        self.log("val_wer", wer_)
        self.log("val_cer", cer_)
        self.log("val_loss", loss_value)
