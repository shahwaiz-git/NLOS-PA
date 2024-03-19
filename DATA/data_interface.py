import os
from os.path import join

import torch
from scipy.io import loadmat

import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor, Normalize


class SimuDataset(Dataset):
    def __init__(self, mixed_signal_dir, direct_signal_dir, target_dir, sensor_mask_dir):
        super().__init__()
        self.mixed_signal_dir = mixed_signal_dir
        self.direct_signal_dir = direct_signal_dir
        self.target_dir = target_dir
        self.sensor_mask_dir = sensor_mask_dir
        self.names = os.listdir(mixed_signal_dir)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        mixed_signal = torch.tensor(loadmat(join(self.mixed_signal_dir, name))['distortion_data'])
        direct_signal = torch.tensor(loadmat(join(self.direct_signal_dir, name))['ground_truth_data'])
        target = ToTensor()(loadmat(join(self.target_dir, name))['pressure'])
        sensor_mask = torch.tensor(loadmat(join(self.sensor_mask_dir, name))['sensor_pos'])
        # TODO: NORM into utils
        mixed_signal = ((mixed_signal - torch.min(mixed_signal,dim=1,keepdim=True).values) /
                        (torch.max(mixed_signal,dim=1,keepdim=True).values - torch.min(mixed_signal,dim=1,keepdim=True).values))

        direct_signal = ((direct_signal - torch.min(direct_signal,dim=1,keepdim=True).values) /
                        (torch.max(direct_signal,dim=1,keepdim=True).values - torch.min(direct_signal,dim=1,keepdim=True).values))
        target /= torch.max(target)
        return mixed_signal, direct_signal, target, sensor_mask, name


class DInterface(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.mixed_signal_dir = kwargs['mixed_signal_dir']
        self.direct_signal_dir = kwargs['direct_signal_dir']
        self.target_dir = kwargs['target_dir']
        self.sensor_mask_dir = kwargs['sensor_mask_dir']
        self.train_size = kwargs['train_size']
        self.val_size = kwargs['val_size']
        self.batch_size = kwargs['batch_size']
        self.num_workers = kwargs['num_workers']
        self.seed = kwargs['seed']

    def setup(self, stage: str) -> None:
        dataset = SimuDataset(self.mixed_signal_dir, self.direct_signal_dir, self.target_dir, self.sensor_mask_dir)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [
            self.train_size, self.val_size, 1-self.train_size-self.val_size],
            generator=torch.Generator().manual_seed(self.seed))

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
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True)
