"""
Convolutional Neural Network (CNN) Architecture with PyTorch Lightning

This module outlines a 1D Convolutional Neural Network designed to classify 
time-series inertial data (accelerometer and gyroscope streams) into 
specific sport activities using automated evaluation metrics.
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import lightning as pl
from sklearn.metrics import precision_score, f1_score

# Direct, clean import from our utilities package
from utils.utility import cm_analysis


class CNN(pl.LightningModule):

    """
    1D CNN Classifier extending PyTorch Lightning for streamlined training,
    validation logging, and automatic confusion matrix extraction.
    """

    def __init__(self, input_dim: int, fold: int, classes_names: list, output_dim: int = 2, learning_rate: float = 1e-3):
        super(CNN, self).__init__()

        self.name = "CNN"
        self.classes_labels = classes_names
        self.fold = fold
        self.classes = output_dim

        # Tracking arrays to cache steps across localized validation and testing epochs
        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []

        self.learning_rate = learning_rate
        self.loss_function = nn.CrossEntropyLoss()

        self.input_dim = 6         # Architectural tracking feature length (acc_x,y,z + gyro_x,y,z)
        self.dim = 64              # Base filters count for features extraction
        self.filter_size = 3       # Kernel dimension for the 1D temporal convolution
        self.window_size = input_dim # Corresponds to sequence samples amount (e.g., 180)

        # Convolutional Block processing temporal shifts across time-series windows
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.window_size, self.dim, self.filter_size),
            nn.BatchNorm1d(self.dim),
            nn.PReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=0.1)
        )

        # Classification dense layer routing compressed data into target output classes
        self.fc1 = nn.Sequential(
            nn.Linear(self.dim * 2, 128),
            nn.PReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def id(self) -> str:
        """Generates a dynamic string identifier for file-naming conventions."""
        return f"{self.name}_{self.window_size}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the internal forward math computational graphs."""
        x = self.conv1(x)
        x = x.view(x.size(0), -1)  # Flatten the output map into a 1D tensor vector
        x = self.fc1(x)
        return x

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Core training loss feedback evaluation executed automatically by Lightning."""
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Core validation metrics validation logic executed at epoch intervals."""
        x, y = batch
        y_hat = self(x)
        val_loss = self.compute_loss(y_hat, y)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Store predictions and ground truths for batch evaluation mapping
        self.val_predictions.append(np.argmax(y_hat.cpu().numpy(), axis=1))
        self.val_targets.append(np.argmax(y.cpu().numpy(), axis=1))

        return val_loss

    def on_validation_epoch_end(self):
        """Assembles validation cache to compute global multi-class metrics."""
        if not self.val_predictions:
            return

        val_predictions = np.concatenate(self.val_predictions)
        val_targets = np.concatenate(self.val_targets)

        precision = precision_score(val_targets, val_predictions, average='macro', zero_division=0)
        f1 = f1_score(val_targets, val_predictions, average='macro', zero_division=0)
        
        self.log("prec_macro", precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("f1_score", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Clear active memory trackers to reset constraints for future epochs
        self.val_predictions.clear()
        self.val_targets.clear()

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Executes terminal test loop tracking metrics over isolated verification datasets."""
        x, y = batch
        y_hat = self(x)
        test_loss = self.compute_loss(y_hat, y)
        self.log("test_loss", test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.test_predictions.append(np.argmax(y_hat.cpu().numpy(), axis=1))
        self.test_targets.append(np.argmax(y.cpu().numpy(), axis=1))

        return test_loss

    def on_test_end(self):
        """Builds and logs confusion matrices locally once evaluation bounds close."""
        output_path = f"output/{self.id()}"
        Path(output_path).mkdir(parents=True, exist_ok=True)

        test_predictions = np.concatenate(self.test_predictions)
        test_target = np.concatenate(self.test_targets)
        
        # Call utility analytical function to persist plotting results
        cm_analysis(
            test_target,
            test_predictions,
            f"{output_path}/confusion_matrix_segments_fold_{self.fold}",
            range(self.classes),
            self.classes_labels,
            specific_title=f"Segments: {self.id()} fold {self.fold}"
        )
        self.fold += 1

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Instantiates structural training hyperparameter tuning optimization nodes."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-6)

    def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Maps default cross entropy equations directly into active computational tracks."""
        return self.loss_function(y_hat, y)