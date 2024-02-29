import numpy
import numpy as np
import scipy
import matplotlib.pyplot as plt
from MODEL.DAS import DAS
from torchvision.transforms import ToTensor
from PIL import Image
from pytorch_lightning import Trainer
from MODEL import MInterface
import torch
import os

sensor_mask_dir = "D:\HISLab\DATASET\StripSkullCT_Simulation\sensor_mask_idx.mat"
model = MInterface.load_from_checkpoint(r"D:\HISLab\DATASET\StripSkullCT_Simulation\RESULT\MODEL\epoch-epoch=93-val_loss-val_loss=0.002-val_PSNR-val_PSNR=26.605-val_SSIM-val_SSIM=0.444.ckpt",
                                        sensor_mask_dir=sensor_mask_dir)

sensor = np.array(scipy.io.loadmat(r"D:\缓存站\matlab.mat")['sensor_data'])
p0 = np.array(scipy.io.loadmat(r"D:\缓存站\p0.mat")['p0'])

sensor_tensor = ToTensor()(sensor).unsqueeze(0).type(torch.float).to('cuda')

with torch.no_grad():
    p0_hat = model(sensor_tensor)

plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
plt.imshow(p0, cmap='jet')
plt.subplot(2,2,3)
plt.imshow(p0_hat[0][0].cpu().numpy(), cmap='jet')
plt.subplot(2,2,4)
plt.imshow(p0_hat[0][1].cpu().numpy(), cmap='jet')
plt.subplot(2,2,2)
plt.imshow(p0_hat.mean(dim=(0,1)).cpu().numpy(), cmap='jet')
plt.colorbar()
plt.show()