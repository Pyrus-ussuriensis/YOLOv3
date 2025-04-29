import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.cfg import cfg


class ConvLayer(nn.Module):
    def __init__(self, channels, in_channels=0):
        super(ConvLayer, self).__init__()
        if in_channels == 0:
            in_channels = channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channels//2, kernel_size=1, stride=1),
            nn.BatchNorm2d(channels//2), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=channels//2, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels), nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.channels = channels
    
    def forward(self, x):
        return self.block(x)

class YOLOLayer(nn.Module):
    def __init__(self, anchors, mask, num_classes, img_dim):
        super().__init__()
        self.device = cfg['device']
        self.anchors = [anchors[i] for i in mask]
        self.num_anchors = len(self.anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_dim = img_dim
    
    def forward(self, x):
        batch_size, _, grid_size, _ = x.shape
        prediction = x.view(
            batch_size, 
            self.num_anchors,
            self.bbox_attrs,
            grid_size,
            grid_size
        ).permute(0,1,3,4,2).contiguous()

        x_centers = torch.sigmoid(prediction[..., 0])
        y_centers = torch.sigmoid(prediction[..., 1])
        pw = prediction[..., 2]
        ph = prediction[..., 3]
        obj_conf = torch.sigmoid(prediction[..., 4])
        class_scores = torch.sigmoid(prediction[..., 5:])


        grid_x = torch.arange(grid_size, device=self.device).repeat(grid_size,1).view([1,1,grid_size,grid_size]).type_as(x)
        grid_y = grid_x.permute(0,1,3,2)

        anchor_w = torch.tensor([a[0] for a in self.anchors], device=self.device).view(1, self.num_anchors, 1,1)
        anchor_h = torch.tensor([a[1] for a in self.anchors], device=self.device).view(1, self.num_anchors, 1,1)

        bx = (x_centers + grid_x) * (self.img_dim / grid_size)
        by = (y_centers + grid_y) * (self.img_dim / grid_size)
        bw = torch.exp(pw) * anchor_w
        bh = torch.exp(ph) * anchor_h

        bx = bx.unsqueeze(-1)
        by = by.unsqueeze(-1)
        bw = bw.unsqueeze(-1)
        bh = bh.unsqueeze(-1)
        obj_conf = obj_conf.unsqueeze(-1)

        pred = torch.cat([bx, by, bw, bh, obj_conf, class_scores], dim=-1)
        pred = pred.view(batch_size, -1 , self.bbox_attrs)
        return pred

class ConvPiles(nn.Module):
    def __init__(self, channels, in_channels):
        super(ConvPiles, self).__init__()
        self.block = nn.Sequential(
            ConvLayer(channels=channels, in_channels=in_channels),
            ConvLayer(channels=channels),
            ConvLayer(channels=channels),
        )
   
    def forward(self, x):
        y = self.block(x)
        return y
    
class Head(nn.Module):
    def __init__(self, channels, in_channels, img_dim, anchors, mask, num_classes):
        super().__init__()
#        anchors = cfg['anchors']
        self.cp = ConvPiles(channels, in_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=255, kernel_size=1, stride=1)
        self.yolo = YOLOLayer(anchors=anchors, mask=mask,num_classes=num_classes, img_dim=img_dim)
    
    def forward(self, x):
        y = self.cp(x)
        y1 = y# 保存输出给后面处理
        y = self.conv1(y)
        z = self.yolo(y)
        return y1, z

class DetectionHead(nn.Module):
    def __init__(self, channels=1024):
        super().__init__()
        anchors = cfg['anchors']
        img_dim = int(cfg['img_dim'])
        self.head1 = Head(channels=channels, in_channels=channels, img_dim=img_dim, anchors=anchors, mask=[6,7,8], num_classes=80)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels//4, kernel_size=1, stride=1),
            nn.BatchNorm2d(channels//4), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.head2 = Head(channels=channels//2, in_channels=768, img_dim=img_dim, anchors=anchors, mask=[3,4,5], num_classes=80)# 768是张量拼接后的形状，如果实时获取，则没有注册，无法在训练时更新
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=channels//2, out_channels=channels//8, kernel_size=1, stride=1),
            nn.BatchNorm2d(channels//8), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
 
        self.head3 = Head(channels=channels//4, in_channels=384, img_dim=img_dim, anchors=anchors, mask=[0,1,2], num_classes=80)# 384


    def forward(self, x, t36, t61):# 这里要取回61, 36层的输出
        y, z1 = self.head1(x)# y 表示模型的第一次预测，z表示预测处理后的结果。
        y = self.block1(y)
        y = torch.cat([y, t61], dim=1)
        y, z2 = self.head2(y)
        y = self.block2(y)
        y = torch.cat([y, t36], dim=1)
        y, z3 = self.head3(y)
        return z1, z2, z3


if __name__ == '__main__':
    from models.backbone import YOLONet
    device = cfg['device']
    input_tensor = torch.randn(1, 3, 320, 320).to(device)
    yolo = YOLONet()
    yolo = yolo.to(device=device)
    output, t36, t61 = yolo(input_tensor)
    detect = DetectionHead()
    detect = detect.to(device)
    z1, z2, z3 = detect(output, t36, t61)
    print(z1.shape, z2.shape, z3.shape)






        