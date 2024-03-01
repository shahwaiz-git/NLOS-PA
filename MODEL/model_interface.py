import torch

import pytorch_lightning as pl

from .DAS import DAS
from .UNet2D import UNet2D
from .UNet1D import UNet1D
from utlis import *


class MInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.das = DAS(sensor_mask_dir=kwargs['sensor_mask_dir'], dt=kwargs['dt'])
        self.unet1d = UNet1D(channels=kwargs['channels'])
        self.unet2d = UNet2D(n_classes=kwargs['n_classes'])
        self.configure_loss()

    def forward(self, mixed):
        direct_signal = self.unet1d(mixed)
        reflected_signal = mixed - direct_signal
        signal = torch.cat((direct_signal.unsqueeze(1), reflected_signal.unsqueeze(1)), dim=1)
        image = self.das(signal)
        direct_image = image[:, 0, :, :].unsqueeze(1)
        reflected_image = image[:, 1, :, :].unsqueeze(1)
        reflected_image = self.unet2d(reflected_image)
        return direct_signal, direct_image + reflected_image

    def training_step(self, batch, batch_idx):
        mixed_signal, direct_signal, target, _ = batch
        signal_hat, image_hat = self.forward(mixed_signal)
        loss1 = self.loss_function1(direct_signal.float(), signal_hat.float())
        loss2 = self.loss_function2(target.float(), image_hat.float())
        loss = loss1 + loss2
        self.log("train loss1", loss1, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log("train loss2", loss2, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log("train loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        return loss

    def evaluate(self, batch, stage):
        assert stage in ['val', 'test']
        mixed_signal, direct_signal, target, names = batch
        signal_hat, image_hat = self.forward(mixed_signal)
        loss1 = self.loss_function1(direct_signal.float(), signal_hat.float())
        loss2 = self.loss_function2(target.float(), image_hat.float())
        loss = loss1 + loss2
        psnr = PSNR(target, image_hat)
        ssim = SSIM(target, image_hat)
        self.log(f"{stage}_loss1", loss1, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log(f"{stage}_loss2", loss2, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log(f"{stage}_PSNR", psnr, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log(f"{stage}_SSIM", ssim, prog_bar=True, batch_size=self.hparams.batch_size)

        if stage == 'test' and self.hparams.save_dir:
            save_result(image_hat, self.hparams.save_dir, names)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')

    def predict_step(self, x):
        signal, y_hat = self.forward(x)
        return signal, y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            self.loss_function1 = torch.nn.MSELoss()
            self.loss_function2 = torch.nn.MSELoss()
        elif loss == 'l1':
            self.loss_function1 = torch.nn.L1Loss()
            self.loss_function2 = torch.nn.L1Loss()
        else:
            raise ValueError("Invalid Loss Type!")