import torch
import torch.nn as nn
import torch.nn.functional as F

class DarknetResidual(nn.Module):
    def __init__(self, channels):
        super(DarknetResidual, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels//2, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=channels//2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=channels//2, out_channels=channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
    
    def forward(self, x):
        y = self.block(x)
        return x + y

class Darknet_53(nn.Module):
    def __init__(self):
        super(Darknet_53, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.convres1 = DarknetResidual(channels=64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.convres2 = nn.Sequential(
            *[DarknetResidual(channels=128) for _ in range(2)]
        )

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.convres3 = nn.Sequential(
            *[DarknetResidual(channels=256) for _ in range(8)]
        )

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.convres4 = nn.Sequential(
            *[DarknetResidual(channels=512) for _ in range(8)]
        )

        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1)
        self.convres5 = nn.Sequential(
            *[DarknetResidual(channels=1024) for _ in range(4)]
        )
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.convres1(y)
        y = self.conv3(y)
        y = self.convres2(y)
        y = self.conv4(y)
        y = self.convres3(y)
        t36 = y
        y = self.conv5(y)
        y = self.convres4(y)
        t61 = y
        y = self.conv6(y)
        y = self.convres5(y)
        return y, t36, t61

if __name__ == '__main__':
    input_tensor = torch.randn(1, 3, 256, 256)
    yolo = Darknet_53()
    output, t36, t61 = yolo(input_tensor)
    print(output.shape, t36.shape, t61.shape)