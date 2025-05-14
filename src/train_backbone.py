# basic
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, Dataset
# parameters
from src.utils.cfg import cfg
# model
from models.Darknet53Classifier import Darknet53Classifier
# data
from data.imagenet_dataset import train_loader_imgn, val_loader_imgn 
# optim
from torch.optim import Adam
# log
from torchmetrics import Accuracy
from src.utils.tensorboard import writer
from utils.log import log_info
import logging
from utils.log import logger
# weights
from utils.weights import load_model, save_model


# train settings
## model
model = Darknet53Classifier()
## other parameters
device = cfg['device']
epochs = int(cfg['epochs'])
freq = int(cfg['freq'])
batch_size = int(cfg['batch_size'])

mean = cfg['mean']
std = cfg['std']

step_size=int(cfg['step_size'])
gamma = float(cfg['gamma'])

metric = Accuracy(task="multiclass", num_classes=1000).to(device)

## optim
lr = float(cfg['lr'])
optimizer = Adam(params=model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# 加载权重，训练，保存权重
load = int(cfg['load'])
save = int(cfg['save'])

# loss_fn
loss_fn = F.cross_entropy

def train( model: nn.Module, train_loader: DataLoader, 
          optimizer: torch.optim.Optimizer, loss_fn, device: torch.device, 
          epoch: int, log_fn, freq, save, metric):
    model = model.to(device)
    model.train()

    for step, batch in enumerate(tqdm(train_loader)):
        images, targets = batch
        images = images.to(device)
        targets = targets.to(device)
        LT = 0 

        optimizer.zero_grad()

        outputs = model(images)
        
        LT = loss_fn(outputs, targets)
        acc = metric(outputs, targets)

        # 优化
        LT.backward()
        optimizer.step()

    # 记录日志，保存权重，验证类似，但不用优化
    log_fn(epoch=epoch, loss=LT.item(), mode='train/loss', place='epoch')
    log_fn(epoch=epoch, loss=acc, mode='train/acc', place='epoch')
    save_model(model, save, name='imgn')


def validate( model: nn.Module, val_loader: DataLoader, 
          optimizer: torch.optim.Optimizer, loss_fn, device: torch.device, 
          epoch: int, log_fn, freq, metric):
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        total_loss = 0
        total_acc = 0
        for step, batch in enumerate(tqdm(val_loader)):
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = loss_fn(outputs, targets)
            total_acc += metric(outputs, targets)
            total_loss += loss.item()

        log_fn(epoch=epoch, loss=total_loss/len(val_loader), mode='val/loss', place='epoch')
        avg_acc = total_acc/len(val_loader)
        log_fn(epoch=epoch, loss=avg_acc, mode='val/acc', place='epoch')
        return avg_acc


if __name__ == '__main__':
    if os.path.isfile(os.path.join('weights/', f'YOLOv3_imgn{load}.pth')):
        load_model(model, load)
    
    avg_acc_best = 0
    for epoch in range(epochs):
        train(model=model, train_loader=train_loader_imgn, optimizer=optimizer, loss_fn=loss_fn, device=device, epoch=epoch, log_fn=log_info, freq=freq, save=save,
                batch_size=batch_size, metric=metric)
        avg_acc = validate(model=model, val_loader=val_loader_imgn, optimizer=optimizer, loss_fn=loss_fn, device=device, epoch=epoch, log_fn=log_info, freq=freq, metric=metric)
        if avg_acc > avg_acc_best:
            avg_acc_best = avg_acc
            save_model(model, save, name='imgn_test')

        scheduler.step()
        logger.info(f'current lr:{scheduler.get_last_lr()[0]}')

    # 停止日志和Tensorboard的记录               
    logging.shutdown()
    writer.close()

