from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),

            nn.Conv1d(in_channels=channels, out_channels=channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            DoubleConv(channels),
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose1d(channels, channels, kernel_size=2, stride=2, padding=0),
            DoubleConv(channels)
        )

    def forward(self, x):
        return self.up(x)


class UNet1D(nn.Module):
    def __init__(self, channels):
        super(UNet1D, self).__init__()
        self.down1 = DoubleConv(channels)
        self.down2 = Down(channels)
        self.down3 = Down(channels)
        self.down4 = Down(channels)
        self.down5 = Down(channels)

        self.up1 = Up(channels)
        self.up2 = Up(channels)
        self.up3 = Up(channels)
        self.up4 = Up(channels)
        self.out = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.down1(x)
        out = self.down2(out)
        out = self.down3(out)
        out = self.down4(out)
        out = self.down5(out)

        out = self.up1(out)
        out = self.up2(out)
        out = self.up3(out)
        out = self.up4(out)

        out = self.out(out)
        return out

# if __name__ == '__main__':
#     import torch
#     net = UNet1D(channels=25)
#     # batch, sensor_num, time
#     input = torch.randn(3, 25, 512)
#     out = net(input)
#     print(out.shape)
