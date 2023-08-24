from utils import AN4, collate_pad, load_data
from torch.utils.data import DataLoader
from jasper import JasperModel
from jiwer import wer, cer
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


def test_model(model):
    test_data = load_data("test")
    test_audio_files = [sample["audio_path"] for sample in test_data.values()]
    ground_truth = [sample["label"] for sample in test_data.values()]

    hypothesis = model.transcribe(test_audio_files, batch_size=32)

    wer_ = wer(ground_truth, hypothesis)
    cer_ = cer(ground_truth, hypothesis)

    print(f"wer: {wer_}")
    print(f"cer: {cer_}")



if __name__ == '__main__':
    # Instantiate the model
    num_classes = 28
    num_mels = 64
    dropout = 0.3

    # need to define dataloader
    train_dataset = AN4(load_data("train"))
    val_dataset = AN4(load_data("val"))
    train_dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_pad)
    val_dataloader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_pad)

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_wer',  # Specify the metric to monitor (e.g., 'val_loss')
        mode='min',  # Specify whether to minimize or maximize the monitored metric
        save_top_k=1,  # Number of best models to keep (1 for the best model)
        save_last=True,  # Save the last checkpoint as well
        filename='best_model'  # Prefix for the saved checkpoint filenames
    )

    model = JasperModel(num_classes, num_mels, dropout=dropout, spec_augment=True)

    # uncomment for using trained model
    # model = model.load_from_checkpoint("trained/model.ckpt", num_classes=num_classes, num_mels=num_mels, dropout=dropout)

    # Instantiate the PyTorch Lightning Trainer
    tb_logger = TensorBoardLogger(save_dir='./model-logs', name='Jasper')
    trainer = pl.Trainer(max_epochs=150, devices=1, accelerator='mps', logger=tb_logger, callbacks=[checkpoint_callback])  # Adjust options as needed

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)




