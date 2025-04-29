from pycocotools.coco import COCO
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader, random_split
from src.utils.cfg import cfg
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

imgs_train = cfg['imgs_train']
imgs_val = cfg['imgs_val']
anns_train = cfg['anns_train']
anns_val = cfg['anns_val']
anns_sub = cfg['anns_sub']

mean = cfg['mean']
std = cfg['std']
batch_size = cfg['batch_size']
pic_size = int(cfg['pic_size'])
mode = cfg['mode']
train_subset_len = int(cfg['train_subset_len'])
val_subset_len = int(cfg['val_subset_len'])


# 预定义管线，格式为 COCO [x,y,w,h]
alb_pipeline = A.Compose(
    [
        A.RandomResizedCrop(pic_size, pic_size, scale=(0.8,1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),  # 转为 Tensor[C,H,W]
    ],
    bbox_params=A.BboxParams(
        format="coco",         # COCO 格式
        label_fields=["labels"], 
        min_visibility=0.3     # 剔除过小框
    )
)

def transforms_coco(image, target):
    # PIL.Image → NumPy
    image_np = np.array(image)
    # 将 bbox 及 labels 列表从 target 取出
    bboxes = target["bbox"]
    labels = target["category_id"]
    # 执行变换
    augmented = alb_pipeline(image=image_np, bboxes=bboxes, labels=labels)
    # 更新
    image = augmented["image"]
    target["bbox"] = augmented["bboxes"]
    target["category_id"] = augmented["labels"]
    return image, target

TrainData = CocoDetection(
    root=imgs_train,
    annFile=anns_train,
    transforms=transforms_coco
)

ValData = CocoDetection(
    root=imgs_val,
    annFile=anns_val,
    transform=transforms_coco
)

SubData = CocoDetection(
    root=imgs_train,
    annFile=anns_sub,
    transform=transforms_coco
)

def detection_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, targets

# 根据模式确定要加载的数据集
if mode == 'train':
    TrainData, _ = random_split(TrainData, [train_subset_len, len(TrainData)-train_subset_len])
    ValData, _ = random_split(ValData, [val_subset_len, len(ValData)-val_subset_len])
elif mode == 'test':
    pass
    batch_size = 1
elif mode == 'full_train':
    pass

TrainLoader = DataLoader(dataset=TrainData, batch_size = batch_size)
ValLoader = DataLoader(dataset=ValData, batch_size = batch_size)
SubLoader = DataLoader(dataset=SubData, batch_size=batch_size)

