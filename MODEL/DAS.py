import numpy as np
import torch
from torch import nn
import scipy


class DAS(nn.Module):
    def __init__(self, sensor_mask_dir, dt,
                 Nx=512, Ny=512,
                 dx=1e-4, dy=1e-4,
                 vs=1550):
        super(DAS, self).__init__()
        sensor_mask = scipy.io.loadmat(sensor_mask_dir)['sensor_mask_idx']
        self.sensor_mask = sensor_mask[:, np.lexsort((sensor_mask[0], sensor_mask[1]))].T

        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy
        self.vs = vs
        self.dt = dt

    def forward(self, sensor_data):
        """
        :param sensor_data: (batch, 2, sensor_number, time)
        the first channel represent the direct sensor_data and the second channel represent the reflected sensor_data
        :return: reconstruct image (batch, 2, Nx, Ny)
        """
        batch = sensor_data.shape[0]
        # TODO: device problem
        image = torch.zeros(batch, 2, self.Nx, self.Ny, device='cuda')

        # att: needn't .copy() here
        idx = torch.arange(1, 513, device='cuda').unsqueeze(1).repeat(batch, 2, 1, 512)  # batch, 2, Nx, Ny
        idy = torch.arange(1, 513, device='cuda').unsqueeze(0).repeat(batch, 2, 512, 1)  # batch, 2, Nx, Ny
        for c, (x, y) in enumerate(self.sensor_mask):
            dis = torch.sqrt(((x - idx + 1) * self.dx) ** 2 + ((y - idy + 1) * self.dy) ** 2)  # batch, 2, Nx, Ny
            t = (dis / self.vs / self.dt).to(torch.long).view(batch, 2, -1)  # batch, 2, Nx*Ny
            image += torch.gather(sensor_data[:, :, c, :], dim=2, index=t).view(batch, 2, self.Nx, self.Ny)
        # image /= torch.max(image)
        return image


if __name__ == '__main__':
    current_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    sensor_mask_dir = r'.\sensor_mask_idx.mat'
    net = DAS(sensor_mask_dir, dt=1 / 5e6)
    input_ = torch.tensor(scipy.io.loadmat(r"D:\HISLab\DATASET\StripSkullCT_Simulation\direct_signal\300_410.mat")['direct_signal'],device='cuda').repeat(1,2,1,1)
    out = net(input_)

    additional_memory = torch.cuda.memory_allocated() - (current_memory)
    peak_memory = torch.cuda.max_memory_allocated()
    additional_peak_memory = peak_memory - (current_memory)

    print(f"Additional memory used: {additional_memory / (1024 ** 3)} GB")
    print(f"Additional peak memory used: {additional_peak_memory / (1024 ** 3)} GB")


    # import matplotlib.pyplot as plt
    # plt.imshow(out[0][0].cpu())
    # plt.show()

