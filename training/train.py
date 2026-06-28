"""
Training Pipeline with K-Fold Cross-Validation

This script initializes the custom PyTorch Lightning DataModule, sets up a 3-fold 
cross-validation loop, and coordinates the training, checkpointing, and testing 
phases of the CNN classifier.
"""

import torch
from CNN import CNN
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from dataset import CSVDataModule

# Optimization tricks to maximize throughput on modern hardware
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True


def main():
    path = "data"
    k = 3                   # Number of split blocks for cross-validation
    window_size = 3         # Window span in seconds
    sampling_frequency = 60 # Sensor sampling rate in Hz
    
    # Calculate total sequence steps (3 seconds * 60 samples/sec = 180 inputs)
    input_sequence_length = window_size * sampling_frequency

    # Initialize the pipeline pipeline data handling engine
    data_module = CSVDataModule(
        root_dir=path,
        batch_size=32,
        k_folds=k,
        window_size=window_size,
    )

    # Parse raw CSV data and generate cross-validation indexing splits
    data_module.setup()

    # Iterate through each statistical fold boundary
    for fold in range(data_module.k):
        # Adjusted display log to represent the real index out of total folds
        print(f'\n--- Starting Fold {fold + 1}/{data_module.k} ---')

        # Instantiate a clean CNN model instance for the current validation fold
        model = CNN(
            input_dim=input_sequence_length, 
            fold=fold + 1, 
            classes_names=data_module.legend
        )

        # Monitor loss and save only the highest performing model parameters on disk
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=f'checkpoints/fold_{fold + 1}',
            filename='best-checkpoint',
            save_top_k=1,
            mode='min'
        )

        # High-level training execution engine abstraction
        trainer = Trainer(
            max_epochs=20,
            accelerator="gpu",   # Toggle automatically to "cpu" if local GPU stack is unavailable
            devices=1,
            logger=True,
            log_every_n_steps=10,
            callbacks=[checkpoint_callback]
        )

        # Trigger optimization loss backpropagation loops
        trainer.fit(
            model, 
            train_dataloaders=data_module.train_dataloader(fold=fold), 
            val_dataloaders=data_module.val_dataloader(fold=fold)
        )

        # Run independent performance check using isolated validation splits
        trainer.test(model, dataloaders=data_module.val_dataloader(fold=fold))


if __name__ == '__main__':
    main()