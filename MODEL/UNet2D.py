from torch import nn
import torch


def crop(x1, x2):
    """
    crop the size of x1 to x2
    """
    x1_size = x1.size()[2:4]
    x2_size = x2.size()[2:4]
    top = (x1_size[0] - x2_size[0]) // 2
    bottom = top + x2_size[0]
    left = (x1_size[1] - x2_size[1]) // 2
    right = left + x2_size[1]
    x1_crop = x1[:, :, top:bottom, left:right]
    return x1_crop


class DoubleConv(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            DoubleConv(in_channels, in_channels * 2),
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
            DoubleConv(in_channels, in_channels//2)
        )

    def forward(self, x):
        return self.up(x)


class UNet2D(nn.Module):
    def __init__(self, n_classes=1):
        super(UNet2D, self).__init__()
        self.down1 = DoubleConv(1, 4)
        self.down2 = Down(4)
        self.down3 = Down(8)
        self.down4 = Down(16)
        self.down5 = Down(32)

        self.up1 = Up(64)
        self.up2 = Up(32)
        self.up3 = Up(16)
        self.up4 = Up(8)
        self.out = nn.Sequential(
            nn.Conv2d(4, out_channels=n_classes, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        x = self.out(x)
        return x

if __name__ == '__main__':
    from torchsummary import summary

    net = UNet2D().to("cuda")

    summary(net, (1, 256, 256), device="cuda", batch_size=64)

