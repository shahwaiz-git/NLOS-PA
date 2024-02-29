import math
import os
from os.path import join

import torch
from scipy.io import loadmat

import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor


class SimuDataset(Dataset):
    def __init__(self, mixed_signal_dir, direct_signal_dir, target_dir):
        super().__init__()
        self.mixed_signal_dir = mixed_signal_dir
        self.direct_signal_dir = direct_signal_dir
        self.target_dir = target_dir
        self.names = os.listdir(mixed_signal_dir)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        mixed_signal = torch.tensor(loadmat(join(self.mixed_signal_dir, name))['mixed_signal'])
        direct_signal = torch.tensor(loadmat(join(self.direct_signal_dir, name))['direct_signal'])
        target = ToTensor()(loadmat(join(self.target_dir, name))['target'])
        target /= torch.max(target)
        return mixed_signal, direct_signal, target, name


class DInterface(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.mixed_signal_dir = kwargs['mixed_signal_dir']
        self.direct_signal_dir = kwargs['direct_signal_dir']
        self.target_dir = kwargs['target_dir']
        self.train_size = kwargs['train_size']
        self.val_size = kwargs['val_size']
        self.batch_size = kwargs['batch_size']
        self.num_workers = kwargs['num_workers']

    def setup(self, stage: str) -> None:
        dataset = SimuDataset(self.mixed_signal_dir, self.direct_signal_dir, self.target_dir)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [
            self.train_size-self.val_size, self.val_size, 1-self.train_size])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
            pin_memory=True)

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
            pin_memory=True)

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            pin_memory=True)
