from models.backbone import Darknet_53
from models.Classifier import Classifier
import torch.nn as nn

class Darknet53Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Darknet_53()
        self.classifier = Classifier()

    def forward(self, x):
        y = self.backbone(x)[0]
        y = self.classifier(y)
        return y
