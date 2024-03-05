from MODEL import MInterface
from DATA import DInterface

from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import Trainer, callbacks
from pytorch_lightning.loggers import WandbLogger

from os.path import join

import torch
torch.set_float32_matmul_precision('high')
# --------------------------------parser-------------------------------
parser = ArgumentParser()
parser.add_argument('--model_name', default='UNet-DAS', type=str)

# Data direction
parser.add_argument('--base_dir', default=r'D:\HISLab\DATASET\StripSkullCT_Simulation', type=str)
parser.add_argument('--mixed_signal_dir', default=None, type=str)
parser.add_argument('--direct_signal_dir', default=None, type=str)
parser.add_argument('--target_dir', default=None, type=str)
parser.add_argument('--sensor_mask_dir', default=None, type=str)
parser.add_argument('--save_dir', default=None, type=str)

# Model Control
parser.add_argument('--n_classes', default=1, type=int)
parser.add_argument('--channels', default=50, type=int)
parser.add_argument('--dt', default=1 / 5e6, type=float)

parser.add_argument('--train_size', default=0.8, type=float)
parser.add_argument('--val_size', default=0.1, type=float)  # included in train_size
parser.add_argument('--max_epochs', default=300, type=int)
parser.add_argument('--min_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_workers', default=10, type=int)
parser.add_argument('--seed', default=1121, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--loss', default='mse', type=str)  # mse or l1

# Other
args = parser.parse_args()
args.mixed_signal_dir = join(args.base_dir, 'mixed_signal')
args.direct_signal_dir = join(args.base_dir, 'direct_signal')
args.target_dir = join(args.base_dir, 'target')
args.sensor_mask_dir = join(args.base_dir, 'sensor_mask_idx.mat')
args.save_dir = join(args.base_dir, 'RESULT')

# --------------------------------callback-------------------------------
callback_checkpoint = callbacks.ModelCheckpoint(
    save_top_k=3,
    save_last=True,
    monitor="val_SSIM",
    mode='max',
    dirpath=join(args.save_dir, 'MODEL'),
    filename="{epoch:02d}-{val_loss:.3f}-{val_PSNR:.3f}-{val_SSIM:.3f}",
    save_weights_only=True,
)

callback_early_stop = callbacks.EarlyStopping(
    monitor="val_SSIM",
    mode='max',
    patience=20,
    verbose=False,
)

# ---------------------------------wandb----------------------------------
# wandb_logger = WandbLogger(project='UNet-DAS')

if __name__ == '__main__':
    pl.seed_everything(args.seed)
    model = MInterface(**vars(args))
    datamodule = DInterface(**vars(args))

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        callbacks=[callback_checkpoint, callback_early_stop],
        # logger=wandb_logger,
    )

    trainer.fit(model=model, datamodule=datamodule)

    # trainer.test(model, ckpt_path=r"D:\HISLab\DATASET\StripSkullCT_Simulation\RESULT\MODEL\last.ckpt",
    #              datamodule=datamodule)

    trainer.test(model, datamodule=datamodule, ckpt_path='best')
