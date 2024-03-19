import numpy as np
import torch
from torch import nn
import scipy


class DAS(nn.Module):
    def __init__(self, dt=1/12.5e6, vs=1500,
                 Nx=256, Ny=256,
                 dx=1e-3, dy=1e-3):
        super(DAS, self).__init__()
        # sensor_mask = scipy.io.loadmat(sensor_mask_dir)['sensor_mask_idx']
        # self.sensor_mask = sensor_mask[:, np.lexsort((sensor_mask[0], sensor_mask[1]))].T

        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy
        self.vs = vs
        self.dt = dt

    def forward(self, sensor_data, sensor_mask):
        """
        :param sensor_data: (batch, sensor_number, time)
        :return: reconstruct image (batch, Nx, Ny)
        """
        batch = sensor_data.shape[0]
        sensor_mask = sensor_mask*1000+128  # batch, sensor num, time
        # TODO: device problem
        image = torch.zeros(batch, self.Nx, self.Ny, device='cuda')

        # att: needn't .copy() here
        idx = torch.arange(1, self.Nx + 1, device='cuda').unsqueeze(1).repeat(batch, 1, self.Nx)  # batch, Nx, Ny
        idy = torch.arange(1, self.Ny + 1, device='cuda').unsqueeze(0).repeat(batch, self.Ny, 1)  # batch, Nx, Ny
        for c, xy in enumerate(sensor_mask.permute(1, 0, 2)):   # xy: batch, 2
            x = xy[:, 0].unsqueeze(1).unsqueeze(1)  # batch,1,1
            y = xy[:, 1].unsqueeze(1).unsqueeze(1)  # batch,1,1
            dis = torch.sqrt(((x - idx + 1) * self.dx) ** 2 + ((y - idy + 1) * self.dy) ** 2)  # batch, Nx, Ny
            t = (dis / self.vs / self.dt).to(torch.long).view(batch, -1)  # batch, Nx*Ny
            image += torch.gather(sensor_data[:, c, :], dim=1, index=t).view(batch, self.Nx, self.Ny)

        image = image.reshape(batch, self.Nx * self.Ny)
        min_v, max_v = torch.min(image, dim=1, keepdim=True).values, torch.max(image, dim=1, keepdim=True).values
        image = (image - min_v) / (max_v - min_v)
        image = image.reshape(batch, self.Nx, self.Ny)
        return image.permute(0,2,1)


if __name__ == '__main__':
    sensor_mask_dir1 = r"D:\HISLab\DATASET\dataset_2D_phantom\sensor_pos\IXI012-HH-1211-T1.mat"
    mask1 = torch.tensor(scipy.io.loadmat(sensor_mask_dir1)['sensor_pos'],device='cuda')
    sensor_mask_dir2 = r"D:\HISLab\DATASET\dataset_2D_phantom\sensor_pos\IXI013-HH-1212-T1.mat"
    mask2 = torch.tensor(scipy.io.loadmat(sensor_mask_dir2)['sensor_pos'],device='cuda')
    mask = torch.stack([mask1, mask2])

    net = DAS(dt=1 / 12.5e6)
    tensor1 = torch.tensor(scipy.io.loadmat(r"D:\HISLab\DATASET\dataset_2D_phantom\ground_truth\IXI012-HH-1211-T1.mat")['ground_truth_data'],device='cuda')
    tensor2 = torch.tensor(scipy.io.loadmat(r"D:\HISLab\DATASET\dataset_2D_phantom\ground_truth\IXI013-HH-1212-T1.mat")['ground_truth_data'],device='cuda')
    tensor = torch.stack([tensor1,tensor2])
    out = net(tensor,mask)

    import matplotlib.pyplot as plt
    for i in range(2):
        plt.imshow(out[i].cpu())
        plt.colorbar()
        plt.title(i)
        plt.show()
