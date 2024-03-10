import numpy
import numpy as np
import scipy
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from PIL import Image
from pytorch_lightning import Trainer
from MODEL import MInterface
import torch

sensor_mask_dir = "D:\HISLab\DATASET\StripSkullCT_Simulation\sensor_mask_idx.mat"
model = MInterface.load_from_checkpoint(r"D:\HISLab\DATASET\StripSkullCT_Simulation\RESULT\MODEL\last.ckpt",
                                        sensor_mask_dir=sensor_mask_dir).to('cuda')

sensor_data = np.array(scipy.io.loadmat(r"D:\HISLab\DATASET\StripSkullCT_Simulation\test\mixed_signal\130150_100170.mat")['sensor_data'])
p0 = np.array(scipy.io.loadmat(r"D:\HISLab\DATASET\StripSkullCT_Simulation\test\target\130150_100170.mat")['p0'])

sensor_data = torch.tensor(sensor_data, device='cuda')
mixed_signal = ((sensor_data - torch.min(sensor_data, dim=1, keepdim=True).values) /
                (torch.max(sensor_data, dim=1, keepdim=True).values - torch.min(sensor_data, dim=1,
                                                                                 keepdim=True).values)).unsqueeze(0)

with torch.no_grad():
    direct_signal_hat, direct_image_hat, reflected_image_hat = model(mixed_signal)

plt.figure(figsize=(10,8))
plt.subplot(1,2,1)
plt.title('target',fontsize=14)
plt.imshow(p0/8,cmap='viridis')

plt.subplot(1,2,2)
plt.title('ours',fontsize=14)
plt.imshow((direct_image_hat+reflected_image_hat)[0][0].cpu(),cmap='viridis')

plt.show()