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
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                     kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.conv = DoubleConv(in_channels, in_channels // 2)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        x1 = crop(x1, x2)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    def __init__(self, n_classes=1):
        super(UNet2D, self).__init__()
        self.down1 = DoubleConv(1, 64)
        self.down2 = Down(64)
        self.down3 = Down(128)
        self.down4 = Down(256)
        self.down5 = Down(512)

        self.up1 = Up(1024)
        self.up2 = Up(512)
        self.up3 = Up(256)
        self.up4 = Up(128)
        self.out = nn.Conv2d(64, out_channels=n_classes, kernel_size=(3, 3), padding=(1,1))

    def forward(self, x):
        out1 = self.down1(x)
        out2 = self.down2(out1)
        out3 = self.down3(out2)
        out4 = self.down4(out3)
        out5 = self.down5(out4)

        out6 = self.up1(out4, out5)
        out7 = self.up2(out3, out6)
        out8 = self.up3(out2, out7)
        out9 = self.up4(out1, out8)

        out = self.out(out9)
        return out

if __name__ == '__main__':
    # net = UNet2D()
    # input = torch.randn(3, 1, 512, 512)
    # out = net(input)
    # print(out.shape)

    from torchsummary import summary
    net = UNet2D().to("cuda")
    summary(net, (1, 512, 512), device="cuda", batch_size=64)

