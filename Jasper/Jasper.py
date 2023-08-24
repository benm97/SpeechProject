import torch
import torch.nn as nn
import pytorch_lightning as pl
from utils import VOCABULARY,  AN4
from nemo.collections.asr.modules import SpectrogramAugmentation
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from nemo.collections.asr.metrics.wer import WER, CTCDecoding
from omegaconf import OmegaConf, DictConfig
from nemo.collections.asr.losses.ctc import CTCLoss
from itertools import groupby
from jiwer import wer, cer

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class JasperBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout, separable, se):
        super(JasperBlock, self).__init__()

        self.residual = nn.Sequential()
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                      padding="same", groups=1 if separable else out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(out_channels, out_channels // 16, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(out_channels // 16, out_channels, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.se = None

    def forward(self, x):
        input = x
        residual = self.residual(input)
        out = self.conv(input)
        if self.se:
            se_weight = self.se(out)
            out = out * se_weight
        return out + residual


class Encoder(pl.LightningModule):
    def __init__(self, num_mels, dropout, separable, se):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(num_mels, 256, kernel_size=11, stride=1, padding="same")
        self.relu1 = nn.ReLU()
        self.blocks = nn.Sequential(
            JasperBlock(256, 128, kernel_size=11, stride=1, dilation=1, dropout=dropout, separable=separable, se=se),
            # JasperBlock(128, 256, kernel_size=13, stride=1, dilation=1, dropout=dropout, separable=separable, se=se),
            JasperBlock(128, 256, kernel_size=15, stride=1, dilation=1, dropout=dropout, separable=separable, se=se),
            # JasperBlock(256, 256, kernel_size=17, stride=1, dilation=1, dropout=dropout, separable=separable, se=se),
            JasperBlock(256, 256, kernel_size=19, stride=1, dilation=1, dropout=dropout, separable=separable, se=se),
            # JasperBlock(256, 256, kernel_size=21, stride=1, dilation=1, dropout=dropout, separable=separable, se=se)
        )

    def forward(self, input_signal):
        encoded = self.conv1(input_signal)
        encoded = self.relu1(encoded)
        encoded = self.blocks(encoded)
        return encoded


class JasperModel(pl.LightningModule):
    def __init__(self, num_classes, num_mels, dropout, spec_augment=False):
        super(JasperModel, self).__init__()
        self.encoder = Encoder(num_mels, dropout, separable=True, se=True)
        self.decoder = nn.Conv1d(256, num_classes, kernel_size=1)
        self.preprocessor = AudioToMelSpectrogramPreprocessor(features=num_mels)
        self.spec_augmentation = SpectrogramAugmentation()
        self.spec_augment = spec_augment
        self.loss = CTCLoss(num_classes - 1, zero_infinity=True, reduction="mean_batch")
        # self.decoding = CTCDecoding(DictConfig({"strategy": "greedy"}), vocabulary=OmegaConf.to_container(OmegaConf.create([" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        #  "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"])))
        # self._wer = WER(decoding=self.decoding)
        self.lr = 0.001

    def forward(self, input_signal, input_signal_length):
        processed_signal, processed_signal_length = self.preprocessor(input_signal=input_signal,
                                                                      length=input_signal_length)

        if self.spec_augment and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded = self.encoder(input_signal=processed_signal)
        log_probs = self.decoder(encoded).transpose(1, 2)
        greedy_predictions = log_probs.argmax(dim=2, keepdim=False)
        return log_probs, processed_signal_length, greedy_predictions

    def transcribe(self, audio_paths, batch_size=4):
        self.eval()
        test_dataset = AN4(audio_paths=audio_paths)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=self.__collate_pad_audio_only,)
        hypothesis = []
        for input_signal, input_lengths in test_dataloader:
            log_probs, encoded_len, predictions = self.forward(input_signal=input_signal, input_signal_length=input_lengths)
            hypothesis.extend(self.__decode(predictions, encoded_len))
        return hypothesis

    def __decode(self, predictions, prediction_lengths):
        hypothesis = []
        for pred, length in zip(predictions, prediction_lengths):
            all_tokens = [VOCABULARY.inverse[int(token)] for token in pred[:length]]
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
        log_probs, encoded_len, predictions = self.forward(input_signal=signal, input_signal_length=signal_len)
        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
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
        log_probs, encoded_len, predictions = self.forward(input_signal=signal, input_signal_length=signal_len)
        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        # Greedy Decoding
        hypothesis = self.__decode(predictions, encoded_len)
        ground_truth = ["".join([VOCABULARY.inverse[int(token)] for token in sentence]) for sentence in transcript]

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
