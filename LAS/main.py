import pytorch_lightning as pl
from utils import AN4, collate_pad, load_data
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from LAS.LAS import LAS
from utils import LAS_VOCABULARY

if __name__ == '__main__':
    num_classes = len(LAS_VOCABULARY)
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

    model = LAS(num_mels, 128, num_classes, spec_augment=True)

    # uncomment for using trained model
    # model = model.load_from_checkpoint("trained/model.ckpt", num_classes=num_classes, num_mels=num_mels, dropout=dropout)

    # Instantiate the PyTorch Lightning Trainer
    tb_logger = TensorBoardLogger(save_dir='./model-logs', name='LAS')
    trainer = pl.Trainer(max_epochs=150, devices=1, accelerator='mps', logger=tb_logger,
                         callbacks=[checkpoint_callback], gradient_clip_val=0.5)  # Adjust options as needed

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)
