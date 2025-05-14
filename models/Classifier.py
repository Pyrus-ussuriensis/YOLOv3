import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # ... Darknet-53 的卷积部分省略 ...
        # 假设 backbone_out_channels = 1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x.shape == (batch, 1024, H, W)
        x = self.avgpool(x)           # -> (batch, 1024, 1, 1)
        x = x.view(x.size(0), -1)     # -> (batch, 1024)
        x = self.fc(x)                # -> (batch, num_classes)
        x = self.softmax(x)           # -> (batch, num_classes)
        return x

if __name__ == '__main__':
    from models.backbone import Darknet_53
    import torch
    input = torch.randn(1,3,256,256)
    backbone = Darknet_53()
    classifier = Classifier()
    output1 = backbone(input)
    output = classifier(output1[0])
    print(output.shape)