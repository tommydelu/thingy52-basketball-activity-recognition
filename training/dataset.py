"""
Dataset Ingestion and Temporal Windowing Pipeline

This module implements a custom PyTorch Lightning DataModule. It reads multi-class 
raw inertial CSV files from disk, segments continuous time-series signals into 
overlapping sliding windows, applies one-hot label encoding, and exposes optimized 
K-Fold DataLoader iterators.
"""

import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold


class CSVDataModule(pl.LightningDataModule):
    """
    Data Management Engine parsing raw telemetry data into mathematical 
    tensors ready for Convolutional Neural Network ingestion.
    """

    def __init__(self, root_dir: str, batch_size: int = 32, num_workers: int = 4, 
                 k_folds: int = 3, window_size: int = 3, overlap: float = 0.5, sample_rate: int = 60):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.k = k_folds

        # Temporal splitting parameters
        self.window_size = window_size   # Frame window span in seconds (e.g., 3s)
        self.overlap = overlap           # Sliding shift ratio (0.5 means 50% overlap windowing)
        self.sample_rate = sample_rate   # Sensor frequency configured in hardware (Hz)

        # Dynamic placeholder containers populated during setup
        self.legend = None               # List storing readable category names
        self.x, self.y = None, None      # Global features and targets numpy arrays

        # Indexing lookup caches tracking cross-validation boundaries
        self.train_indices = []
        self.val_indices = []

    def windowing(self, data: pd.DataFrame) -> tuple:
        """
        Slices continuous dataframe logs into uniform overlapping sequence matrix windows.
        """
        windows, labels = [], []
        
        # Calculate step strides and absolute sequence sizes based on frequency metrics
        total_window_samples = self.window_size * self.sample_rate
        stride_step = int(self.overlap * self.sample_rate)

        # Loop and shift boundary indices across the continuous row logs
        for i in range(0, len(data) - total_window_samples, stride_step):
            window = data.iloc[i:i + total_window_samples]
            
            # Extract categorical label assuming uniform behavior across the window timeframe
            label = window["label"].iloc[0]
            window = window.drop(columns=["label"])
            
            # Cache the extracted values matrix chunk
            windows.append(window.values)
            labels.append(label)

        return np.array(windows), np.array(labels)

    def create_dataset(self) -> tuple:
        """
        Scans directories, parses valid CSV tables, applies windowing transformations, 
        and vectorizes targets.
        """
        x_global, y_global = [], []
        
        for file in os.listdir(self.root_dir):
            if file.endswith(".csv"):
                # Load CSV ignoring time columns, mapping custom names to axes signals
                df = pd.read_csv(
                    os.path.join(self.root_dir, file),
                    index_col=0,
                    header=None,
                    names=["time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
                )
                
                # Derive target string directly from filename mapping structure (e.g., "run_balling.csv")
                df["label"] = file.split("_")[1].split(".")[0]
                
                # Extract structured segmentation segments
                x_segments, y_segments = self.windowing(df)
                x_global.extend(x_segments)
                y_global.extend(y_segments)

        x_global, y_global = np.array(x_global), np.array(y_global)

        # Convert textual target labels into unified one-hot binary distributions
        self.legend, y_encoded = self.labels_encoding(y_global)

        return x_global, y_encoded

    @staticmethod
    def labels_encoding(y: np.ndarray) -> tuple:
        """
        Transforms categorical string rows into numerical one-hot encoded probability arrays.
        """
        categories, inverse_indices = np.unique(y, return_inverse=True)

        # Build an orthogonal binary flag structure matrix mapping active classes
        one_hot = np.zeros((y.size, categories.size))
        one_hot[np.arange(y.size), inverse_indices] = 1
        
        return categories, one_hot.astype(np.float64)

    def setup(self, stage: str = None) -> None:
        """
        Executes parsing logic and pre-calculates validation indices boundaries 
        across global arrays.
        """
        self.x, self.y = self.create_dataset()
        
        # Configure deterministic shuffling strategy to align reproducible training steps
        k_fold = KFold(n_splits=self.k, shuffle=True, random_state=42)

        # Populate internal fold indexing limits maps
        for train_idx, val_idx in k_fold.split(self.x):
            self.train_indices.append(train_idx)
            self.val_indices.append(val_idx)

    def prepare_dataset(self, indexes: list, fold: int) -> TensorDataset:
        """
        Converts indexed numpy allocations into explicit PyTorch tensor graph datatypes.
        """
        x_tensor = torch.tensor(self.x[indexes[fold]], dtype=torch.float32)
        y_tensor = torch.tensor(self.y[indexes[fold]], dtype=torch.float32)
        return TensorDataset(x_tensor, y_tensor)

    def train_dataloader(self, fold: int = 0) -> DataLoader:
        """Generates training data iteration feeds optimized for target cross-validation loops."""
        return DataLoader(
            self.prepare_dataset(self.train_indices, fold),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True  # Keeps background workers alive between training loops
        )

    def val_dataloader(self, fold: int = 0) -> DataLoader:
        """Generates validation data iteration feeds optimized for target evaluation loops."""
        return DataLoader(
            self.prepare_dataset(self.val_indices, fold),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True
        )