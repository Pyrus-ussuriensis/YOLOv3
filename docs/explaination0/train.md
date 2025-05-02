# train.py
## import
```python
# basic
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import os
# parameters
from src.utils.cfg import cfg
# model
from models.YOLOv3 import YOLOv3
# data
from torch.utils.data import DataLoader, Dataset
from data.coco_dataset import TrainLoader, ValLoader, SubLoader
# optim
from torch.optim import Adam
# log
from src.utils.tensorboard import writer
from utils.log import log_info
import logging
# weights
from utils.weights import load_model, save_model
# visualize
from utils.save import save_tensor_box
# cal
from torchvision.ops import box_iou
```
我们要导入pytorch常用的如torch,torchvision,nn,F,进度条tqdm，os等
为了获取train的一系列参数导入cfg
获取模型
获取数据加载器和类
获取Adam优化器
获取日志记录函数，记录日志和Tensorboard
加载和保存权重文件
可视化标准和预测结果
导入计算IoU的函数

## parameters
```python
# train settings
## model
model = YOLOv3()
## other parameters
device = cfg['device']
epochs = int(cfg['epochs'])
freq = int(cfg['freq'])
batch_size = int(cfg['batch_size'])

mean = cfg['mean']
std = cfg['std']

num_classes = cfg['num_classes']
ignore_thresh = float(cfg['ignore_thresh'])
truth_thresh = float(cfg['truth_thresh'])

## optim
lr = float(cfg['lr'])
optimizer = Adam(params=model.parameters(), lr=lr)
## loss
loss_fn = F.mse_loss
## visualize
pic_num = 0

# 加载权重，训练，保存权重
load = int(cfg['load'])
save = int(cfg['save'])
```
设置一些参数，包括模型，优化器
导入一系列超参数，设备等

## get_targets
```python
def get_targets(targets, device, batch_size):
    all_boxes, all_labels = [], []
    for img_i, ann_list in enumerate(targets):
        tmp_boxes = []
        tmp_labels = []
        for ann in ann_list:
            tmp_boxes.append(ann['bbox'])
            tmp_labels.append(ann['category_id'])
        if tmp_boxes == []:
            tmp_boxes = torch.empty(0,4).to(device)
            tmp_labels = torch.empty(0,1).to(device)
        else:    
            tmp_boxes = torch.stack(tmp_boxes)
            tmp_labels = torch.stack(tmp_labels)
        all_boxes.append(tmp_boxes)
        all_labels.append(tmp_labels)
    #boxes   = torch.tensor(all_boxes, dtype=torch.float32, device=device)
    #labels  = torch.tensor(all_labels, dtype=torch.long,    device=device)
    #boxes = boxes.view(batchsize, -1, 4)
    #labels = labels.view(batchsize, -1, 1)
    return all_boxes, all_labels

```
在数据集处理阶段我将图片整合成张量，而标注则是列表里面套字典，字典里是bbox和类别。这个函数的作用是将标注按每个图片的整合成一个张量，然后一个batch的放到一个列表中，这个处理也可以放到数据集文件中进行。
标注是可能是空的，此时加入空张量，如果不是空的就用torch.stack增加一个维度连接所有框的张量，类别也一样，前者变成[N,4],后者变成[N]。

## 根据预测和标注得到损失
```python
def IoU_Masks_Loss(box, label, pred_box, pred_conf, ignore_thresh, truth_thresh, pred_cl, num_classes):
    ious = box_iou(pred_box, box)
    max_iou, best_gt = ious.max(dim=1)              # 对每个预测框找到最大的IoU标注框
    pos_mask = max_iou > truth_thresh
    neg_mask = max_iou <= ignore_thresh
    loss_box = F.mse_loss(pred_box[pos_mask], box[best_gt[pos_mask]])
    loss_obj = F.binary_cross_entropy(pred_conf[pos_mask], torch.ones_like(pred_conf[pos_mask]))
    loss_noobj = F.binary_cross_entropy(pred_conf[neg_mask], torch.zeros_like(pred_conf[neg_mask]))
    cls_target = F.one_hot(label[best_gt[pos_mask]].long(), num_classes=num_classes).float()
    loss_cl = F.binary_cross_entropy_with_logits(pred_cl[pos_mask], cls_target)
    return loss_box, loss_obj, loss_noobj, loss_cl

```
pred_box是[N,4],box是[M,4]，利用函数直接得到一个二维列表，元素为两个框的IoU。然后获得每个预测框最大的IoU的标注框，然后根据truth_thresh得到正样本掩码，同理得到负样本掩码。
正样本要用预测框和标注计算MSE损失，然后正负样本计算置信度损失，分别是全1，全0的BCE。正样本类别计算BCE。

## 训练函数
```python
def train( model: nn.Module, train_loader: DataLoader, 
          optimizer: torch.optim.Optimizer, loss_fn, device: torch.device, 
          epoch: int, log_fn, freq, save, batch_size, num_classes, ignore_thresh=0.7, truth_thresh=1,
            p_box=1.0, p_obj=1.0, p_noobj=0.5, p_class=1.0):
    model = model.to(device)
    model.train()
    global pic_num 

    for step, batch in enumerate(tqdm(train_loader)):
        images, targets = batch
        images = images.to(device)
        LT = 0 

        boxes, labels = get_targets(targets=targets, batch_size=batch_size, device=device) # [[],[]]

        optimizer.zero_grad()

        z1, z2, z3 = model(images)


        for outputs in [z1, z2, z3]:
            pred_cls = outputs[..., 5:5+80]  # (B,A,80)
            pred_boxes = outputs[..., :4] # (B,A,4)
            pred_confs = outputs[..., 4] # (B,A,1)

            len_batch = len(boxes)
            for b in range(len_batch): # 每个batch的长度不同所以单独处理
                box = boxes[b]
                label = labels[b]
                pred_box = pred_boxes[b,...].squeeze(0)
                pred_conf = pred_confs[b,...].squeeze(0)
                pred_cl = pred_cls[b,...].squeeze(0)

                if box.numel() == 0:  
                # 没有任何真实框，跳过回归与分类损失，仅计算 noobj 损失或直接 continue
                    continue
                loss_box, loss_obj, loss_noobj, loss_cl = IoU_Masks_Loss(box, label, pred_box, 
                pred_conf, ignore_thresh, truth_thresh, pred_cl, num_classes)

                LT += p_box*loss_box + p_obj*loss_obj + p_noobj*loss_noobj + p_class*loss_cl

                # 隔50步记录日志，打印图片
                if step % freq == 0:
                    #log_fn(epoch=step, loss=LT, mode='train', place='step')

                    save_tensor_box(image=images[b].squeeze(0), targets=box, label=label, outputs=pred_box, pred_cl=pred_cl, pred_conf=pred_conf, pic_num=pic_num)
                    pic_num += 1

        # 优化
        LT.backward()
        optimizer.step()

    # 记录日志，保存权重，验证类似，但不用优化
    log_fn(epoch=epoch, loss=LT.item(), mode='train', place='epoch')
    save_model(model, save)
```
在每个尺度和每个图片上进行损失的计算，因为不同尺度预测张量的形状不同，不同图片标注的形状不同。

## 验证
```python
def validate( model: nn.Module, val_loader: DataLoader, 
          optimizer: torch.optim.Optimizer, loss_fn, device: torch.device, 
          epoch: int, log_fn, freq, batch_size, num_classes, ignore_thresh=0.7, truth_thresh=1,
            p_box=1.0, p_obj=1.0, p_noobj=0.5, p_class=1.0):
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        total_loss = 0
        for step, batch in enumerate(tqdm(val_loader)):
            images, targets = batch
            images = images.to(device)

            boxes, labels = get_targets(targets, batch_size, device) # [[],[]]

            optimizer.zero_grad()

            z1, z2, z3 = model(images)


            for outputs in [z1, z2, z3]:
                pred_cls = outputs[..., 5:5+80]  # (B,A,80)
                pred_boxes = outputs[..., :4] # (B,A,4)
                pred_confs = outputs[..., 4] # (B,A,1)

                for b in range(batch_size): # 每个batch的长度不同所以单独处理
                    box = boxes[b]
                    label = labels[b]
                    pred_box = pred_boxes[b,...].squeeze(0)
                    pred_conf = pred_confs[b,...].squeeze(0)
                    pred_cl = pred_cls[b,...].squeeze(0)
                    loss_box, loss_obj, loss_noobj, loss_cl = IoU_Masks_Loss(box, label, pred_box, 
                    pred_conf, ignore_thresh, truth_thresh, pred_cl, num_classes)

                    total_loss += p_box*loss_box.item() + p_obj*loss_obj.item() + p_noobj*loss_noobj.item() + p_class*loss_cl.item()


        log_fn(epoch=epoch, loss=total_loss/len(val_loader), mode='val', place='epoch')

```
同上

## 训练
```python
if __name__ == '__main__':
    if os.path.isfile(os.path.join('weights/', f'RTST_{load}.pth')):
        load_model(model, load)
    
    for epoch in range(epochs):
        train(model=model, train_loader=ValLoader, optimizer=optimizer, loss_fn=loss_fn, device=device, epoch=epoch, log_fn=log_info, freq=freq, save=save,
               num_classes=num_classes, ignore_thresh=ignore_thresh, truth_thresh=truth_thresh, batch_size=batch_size)
        #validate(model=model, val_loader=ValLoader, optimizer=optimizer, loss_fn=loss_fn, device=device, epoch=epoch, log_fn=log_info, freq=freq,
        #           num_classes=num_classes, ignore_thresh=ignore_thresh, truth_thresh=truth_thresh)
    save_model(model, save)

    # 停止日志和Tensorboard的记录               
    logging.shutdown()
    writer.close()

```
获取权重，训练验证，保存权重，停止记录。