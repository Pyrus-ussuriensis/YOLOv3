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
from models.YOLOv3 import YOLOv3
# data
from data.coco_dataset import TrainLoader, ValLoader, SubLoader
# optim
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
# weights
from utils.weights import load_model, save_model
# visualize
from utils.save import save_tensor_box
# cal
from torchvision.ops import box_iou, box_convert
# CLI
import argparse, yaml

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv3 train")

    parser.add_argument('--experiment', '-e', type=int, help='number of experiment')
    parser.add_argument("--load", "-l", type=int, help='the num of loading weight files')
    parser.add_argument("--save", "-s", type=int, help='the num of saving weight files')
    parser.add_argument("--batchsize", "-b", type=int, default=16, help='batchsize')
    parser.add_argument("--mode", "-m", type=str, choices=['test', 'train', 'full_train'], default='full_train', help='training mode')
    parser.add_argument("--lr", type=int, default=1e-3, help='learning rate')
    parser.add_argument("--momentum", type=int, default=0.9, help='momentum')
    parser.add_argument("--weight", '-w', type=int, default=0.0005, help='weight decay')
    parser.add_argument("--epoch", type=int, default=300, help='num of epochs')
    parser.add_argument("--step", type=int, default=30, help='num of step size')
    parser.add_argument("--gamma", '-g', type=int, default=0.1, help='gamma')
    return parser.parse_args()

args = parse_args()
if args.experiment is not None: cfg['experiment'] = args.experiment 
if args.load is not None: cfg['load'] = args.load
if args.save is not None: cfg['save'] = args.save
if args.batchsize is not None: cfg['batchsize'] = args.batchsize
if args.mode is not None: cfg['mode'] = args.mode
if args.lr is not None: cfg['lr'] = args.lr 
if args.momentum is not None: cfg['momentum'] = args.momentum
if args.weight is not None: cfg['weight'] = args.weight
if args.epoch is not None: cfg['epoch'] = args.epoch
if args.step is not None: cfg['step'] = args.step
if args.gamma is not None: cfg['gamma'] = args.gamma

# log
from src.utils.tensorboard import writer
from utils.log import log_info
import logging


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
conf_thresh = float(cfg['conf_thresh'])

## optim
lr = float(cfg['lr'])
momentum = float(cfg['momentum'])
weight_decay = float(cfg['weight_decay'])
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
scheduler = CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-5)
#optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
## loss
loss_fn = F.mse_loss
## visualize
pic_num = 0

# 加载权重，训练，保存权重
load = int(cfg['load'])
save = int(cfg['save'])

# ap
coco_result = cfg['coco_result']
anns_ap = cfg['anns_ap']


def get_targets(targets, device, batch_size):
    targets = [ann_list[1:] for ann_list in targets]
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



# box，label是标注，pred_box,pred_conf,pred_cl是预测，余下是阈值，类别数，设备
def IoU_Masks_Loss(box, label, pred_box, pred_conf, ignore_thresh, truth_thresh, pred_cl, num_classes, device):
    pred_box_xyxy = box_convert(pred_box, in_fmt='cxcywh', out_fmt='xyxy')
    box_xyxy = box_convert(box, in_fmt='xywh', out_fmt='xyxy')
    box_cxcywh = box_convert(box, in_fmt='xywh', out_fmt='cxcywh')
    
    # 惩罚小框
    gt_w_abs = box_cxcywh[..., 2]
    gt_h_abs = box_cxcywh[..., 3]
    img_dim_float = float(cfg['img_dim']) # Make sure img_dim is available, e.g. via cfg

    gt_w_normalized = gt_w_abs / img_dim_float
    gt_h_normalized = gt_h_abs / img_dim_float

    box_loss_scale = 2.0 - gt_w_normalized * gt_h_normalized
    box_loss_scale = box_loss_scale.clamp(min=0.1) # Ensure scale is not too small or negative


    ious = box_iou(pred_box_xyxy, box_xyxy)
    max_iou, best_gt = ious.max(dim=1)              # 对每个预测框找到最大的IoU标注框
    _, pos_gt = ious.max(dim=0) # 对每个标注框找到最大IoU为正样本
    pos_mask = torch.zeros_like(max_iou, dtype=torch.bool, device=device)
    pos_mask[pos_gt] = True
    #pos_mask.scatter_(0, pos_gt, True)
    #pos_mask = max_iou > truth_thresh
    neg_mask = max_iou <= ignore_thresh
    neg_mask.masked_fill_(pos_mask, False) 

    loss_box = box_loss_scale.unsqueeze(-1) * F.mse_loss(pred_box[pos_gt], box_cxcywh)
    loss_box = loss_box.mean()
    loss_obj = F.binary_cross_entropy(pred_conf[pos_mask], torch.ones_like(pred_conf[pos_mask]))
    loss_noobj = F.binary_cross_entropy(pred_conf[neg_mask], torch.zeros_like(pred_conf[neg_mask]))
    ### 类别不是连续，要修改。
    mapping = cfg['mapping']
    label_ctg = mapping[label.long()]
    cls_target = F.one_hot(label_ctg[best_gt[pos_mask]], num_classes=num_classes).float()
    loss_cl = F.binary_cross_entropy(pred_cl[pos_mask], cls_target)

    #if signal:
    #    save_tensor_box(image=image, targets=box, label=label, outputs=pred_box[pos_mask], pred_cl=pred_cl[pos_mask], 
    #                                pred_conf=pred_conf[pos_mask], pic_num=pic_num)

    return loss_box, loss_obj, loss_noobj, loss_cl


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
        L_box, L_obj, L_noobj, L_cl = 0,0,0,0

        boxes, labels = get_targets(targets=targets, batch_size=batch_size, device=device) # [[],[]]

        optimizer.zero_grad()

        z1, z2, z3 = model(images)


        for outputs in [z1, z2, z3]:
            pred_cls = outputs[..., 5:5+80]  # (B,A,80)
            pred_boxes = outputs[..., :4] # (B,A,4)
            pred_confs = outputs[..., 4] # (B,A,1)
            #y_cls = outputs[0][..., 5:5+80]  # (B,A,80) 
            #y_boxes = outputs[0][..., :4] # (B,A,4)

            len_batch = len(boxes)
            for b in range(len_batch): # 每个batch的长度不同所以单独处理
                box = boxes[b]
                label = labels[b]
                pred_box = pred_boxes[b,...].squeeze(0)
                pred_conf = pred_confs[b,...].squeeze(0)
                pred_cl = pred_cls[b,...].squeeze(0)
            #    y_box = y_boxes[b,...].squeeze(0)
            #    y_cl = y_cls[b,...].squeeze(0)

                if box.numel() == 0:  
                # 没有任何真实框，跳过回归与分类损失，仅计算 noobj 损失或直接 continue
                    #neg_mask = torch.ones_like(max_iou, dtype=torch.bool, device=device)
                    loss_noobj = F.binary_cross_entropy(pred_conf, torch.zeros_like(pred_conf))
                    LT += p_noobj * loss_noobj
                    continue

                #signal = False
                # 隔50步记录日志，打印图片
                #freq = 1
                if epoch % freq == 0 and step == 0:
                    #log_fn(epoch=step, loss=LT, mode='train', place='step')

                    save_tensor_box(image=images[b].squeeze(0), targets=box, label=label, outputs=pred_box, pred_cl=pred_cl, 
                                    pred_conf=pred_conf, pic_num=pic_num)
                    pic_num += 1
                    #signal = True

                loss_box, loss_obj, loss_noobj, loss_cl = IoU_Masks_Loss(box, label, pred_box, 
                pred_conf, ignore_thresh, truth_thresh, pred_cl, num_classes, device=device)

                L_box += loss_box.item()
                L_obj += loss_obj.item()
                L_noobj += loss_noobj.item()
                L_cl += loss_cl.item()

                LT += p_box*loss_box + p_obj*loss_obj + p_noobj*loss_noobj + p_class*loss_cl

                

        # 优化
        LT.backward()
        optimizer.step()

    # 记录日志，保存权重，验证类似，但不用优化
    log_fn(epoch=epoch, loss=LT.item(), mode='train', place='epoch')
    log_fn(epoch=epoch, loss=L_box, mode='train/box', place='epoch')
    log_fn(epoch=epoch, loss=L_obj, mode='train/obj', place='epoch')
    log_fn(epoch=epoch, loss=L_noobj, mode='train/noobj', place='epoch')
    log_fn(epoch=epoch, loss=L_cl, mode='train/cl', place='epoch')
    save_model(model, save, name='yolo')


def validate( model: nn.Module, val_loader: DataLoader, 
          optimizer: torch.optim.Optimizer, loss_fn, device: torch.device, 
          epoch: int, log_fn, freq, batch_size, num_classes, ignore_thresh=0.7, truth_thresh=1, conf_thresh=0.5, 
            p_box=1.0, p_obj=1.0, p_noobj=1, p_class=1.0):
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        total_loss = 0
        L_box, L_obj, L_noobj, L_cl = 0,0,0,0
        global pic_num
        for step, batch in enumerate(tqdm(val_loader)):
            images, targets = batch
            images = images.to(device)

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
                        continue
                    #save_tensor_box(image=images[b].squeeze(0), targets=box, label=label, outputs=pred_box, pred_cl=pred_cl, 
                    #                pred_conf=pred_conf, pic_num=pic_num)
                    #pic_num += 1
                    
                    loss_box, loss_obj, loss_noobj, loss_cl = IoU_Masks_Loss(box, label, pred_box, 
                    pred_conf, ignore_thresh, truth_thresh, pred_cl, num_classes, device=device)

                    L_box += loss_box.item()
                    L_obj += loss_obj.item()
                    L_noobj += loss_noobj.item()
                    L_cl += loss_cl.item()
                    total_loss += p_box*loss_box.item() + p_obj*loss_obj.item() + p_noobj*loss_noobj.item() + p_class*loss_cl.item()


        log_fn(epoch=epoch, loss=total_loss/len(val_loader), mode='val', place='epoch')
        log_fn(epoch=epoch, loss=L_box/len(val_loader), mode='val/box', place='epoch')
        log_fn(epoch=epoch, loss=L_obj/len(val_loader), mode='val/obj', place='epoch')
        log_fn(epoch=epoch, loss=L_noobj/len(val_loader), mode='val/noobj', place='epoch')
        log_fn(epoch=epoch, loss=L_cl/len(val_loader), mode='val/cl', place='epoch')

        # === COCO API 评估开始 ===
        if epoch % freq != 1:
            return
        import json
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        # 1. 收集所有预测结果
        coco_results = []
        for step, batch in enumerate(tqdm(val_loader)):
            images, targets = batch
            # 假设 targets 是 [ [ann,ann,…], […], … ]，ann 中含 image_id
            image_ids = [ann_list[0]['image_id'] for ann_list in targets]
            #targets = [ann_list[1:] for ann_list in targets]

            with torch.no_grad():
                z1, z2, z3 = model(images.to(device))
            # 同训练时一样，将 z1/z2/z3 合并然后做 NMS
            # 这里直接复用你 save_tensor_box 中的流程：
            all_outputs = torch.cat([z[..., :4] for z in (z1,z2,z3)], dim=1)
            all_confs   = torch.cat([z[..., 4:5] for z in (z1,z2,z3)], dim=1).squeeze(-1)
            all_cls     = torch.cat([z[..., 5:5+80] for z in (z1,z2,z3)], dim=1)

            for b, image_id in enumerate(image_ids):
                # 1) 计算最终分数并筛选
                cls_conf, cls_idx = all_cls[b].max(dim=1)
                scores = all_confs[b] * cls_conf
                mask   = scores >= conf_thresh
                boxes  = box_convert(all_outputs[b][mask], in_fmt='cxcywh', out_fmt='xywh')
                scores = scores[mask]
                labels = cls_idx[mask]

                # 2) NMS
                from torchvision.ops import batched_nms
                keep = batched_nms(
                    box_convert(boxes, in_fmt='cxcywh', out_fmt='xywh'),
                    scores, labels,
                    iou_threshold=0.45
                ).tolist()
                boxes  = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                # 3) 转为 COCO JSON 条目
                for (x,y,w,h), s, c in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                    coco_results.append({
                        "image_id": int(image_id),
                        "category_id": int(c),      # 与 instances_val.json 中一致
                        "bbox": [x, y, w, h],
                        "score": float(s)
                    })

        # 4. 写入文件并调用 COCOeval
        
        with open(coco_result, 'w') as f:
            json.dump(coco_results, f)

        cocoGt  = COCO(anns_ap)          # GT JSON :contentReference[oaicite:0]{index=0}
        cocoDt  = cocoGt.loadRes(coco_result)              # 预测结果 JSON :contentReference[oaicite:1]{index=1}
        cocoEval= COCOeval(cocoGt, cocoDt, iouType='bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()  # 打印 AP@[.5:.95]、AP@.5、AP@.75 等指标
        # === COCO API 评估结束 ===
    return 


if __name__ == '__main__':
    if os.path.isfile(os.path.join('weights/', f'YOLOv3_{load}.pth')):
        load_model(model, load, 'yolo')
    for epoch in range(epochs):
        train(model=model, train_loader=TrainLoader, optimizer=optimizer, loss_fn=loss_fn, device=device, epoch=epoch, log_fn=log_info, freq=freq, save=save,
              num_classes=num_classes, ignore_thresh=ignore_thresh, truth_thresh=truth_thresh, batch_size=batch_size)
        validate(model=model, val_loader=SubLoader, optimizer=optimizer, loss_fn=loss_fn, device=device, epoch=epoch, log_fn=log_info, freq=freq,
                   batch_size=batch_size, num_classes=num_classes, ignore_thresh=ignore_thresh, truth_thresh=truth_thresh, conf_thresh=conf_thresh)
        scheduler.step()
    save_model(model, save, 'yolo')

    # 停止日志和Tensorboard的记录               
    logging.shutdown()
    writer.close()

