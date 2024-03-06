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

        self.automatic_optimization = False

    def forward(self, mixed_signal):
        direct_signal = self.unet1d(mixed_signal)
        reflected_signal = mixed_signal - direct_signal

        direct_image = self.das(direct_signal.detach()).unsqueeze(1)
        reflected_image = self.das(reflected_signal.detach()).unsqueeze(1)

        reflected_image = self.unet2d(reflected_image)

        return direct_signal, direct_image, reflected_image

    def training_step(self, batch, batch_idx):
        (opt1, opt2) = self.optimizers(use_pl_optimizer=False)

        mixed_signal, direct_signal, target, _ = batch

        direct_signal_hat, direct_image_hat, reflected_image_hat = self.forward(mixed_signal)
        loss1 = self.loss_function1(direct_signal.float(), direct_signal_hat.float())
        opt1.zero_grad()
        self.manual_backward(loss1)
        opt1.step()

        loss2 = self.loss_function2((target-direct_image_hat).float(), reflected_image_hat.float())
        opt2.zero_grad()
        self.manual_backward(loss2)
        opt2.step()

        self.log_dict({'train loss1': loss1, 'train loss2': loss2, 'train loss': loss1 + loss2},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)

    def evaluate(self, batch, stage):
        assert stage in ['val', 'test']
        mixed_signal, direct_signal, target, names = batch
        direct_signal_hat, direct_image_hat, reflected_image_hat = self.forward(mixed_signal)
        image_hat = direct_image_hat + reflected_image_hat
        loss1 = self.loss_function1(direct_signal.float(), direct_signal_hat.float())
        loss2 = self.loss_function2((target-direct_image_hat).float(), reflected_image_hat.float())
        psnr = PSNR(target, image_hat)
        ssim = SSIM(target, image_hat)
        self.log_dict({f"{stage} loss1": loss1, f'{stage} loss2': loss2, f'{stage} loss': loss1 + loss2,
                       f"{stage}_PSNR": psnr, f"{stage}_SSIM": ssim},
                      on_step=False, prog_bar=True, on_epoch=True, batch_size=self.hparams.batch_size)

        if stage == 'test' and self.hparams.save_dir:
            save_image(image_hat, self.hparams.save_dir, names)
            save_mat(direct_signal_hat, self.hparams.save_dir, names)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')

    def predict_step(self, x):
        signal, y_hat = self.forward(x)
        return signal, y_hat

    def configure_optimizers(self):
        opt1 = torch.optim.Adam(self.unet1d.parameters(), lr=self.hparams.lr1)
        opt2 = torch.optim.Adam(self.unet2d.parameters(), lr=self.hparams.lr2)
        return opt1, opt2

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
