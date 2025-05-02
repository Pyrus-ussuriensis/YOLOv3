from models.backbone import Darknet_53
from models.detectionheads import DetectionHead
import torch.nn as nn
import torch
from src.utils.cfg import cfg

class YOLOv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.darknet = Darknet_53()
        self.dhs = DetectionHead()
    
    def forward(self, x):
        z1, z2, z3 = self.dhs(*(self.darknet(x)))
        return z1, z2, z3


if __name__ == '__main__':
    device = cfg['device']
    tensor_in = torch.randn([1,3,320,320]).to(device)
    yolo = YOLOv3().to(device)
    z1, z2, z3 = yolo(tensor_in)
    print(z1.shape, z2.shape, z3.shape)