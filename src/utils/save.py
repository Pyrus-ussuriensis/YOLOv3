from torchvision import transforms
from torchvision.utils import save_image
from src.utils.log import logger
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from pycocotools.coco import COCO
from src.utils.cfg import cfg
from torchvision.ops import box_convert, nms, batched_nms

# 指定 COCO 注释文件路径
anns_train = cfg['anns_train']
mean = cfg['mean']
std = cfg['std']
device = cfg['device']

# 初始化 COCO 对象（会加载并索引 JSON）
coco = COCO(anns_train)

# 获取所有类别 ID，然后加载对应的类别信息
cat_ids = coco.getCatIds()                  # e.g. [1, 2, 3, …, 90] :contentReference[oaicite:0]{index=0}
cats    = coco.loadCats(cat_ids)            # 列表，每项如 {'id':1,'name':'person',…}
cat_id_to_incontiguous = {idx : orig_id for idx, orig_id in enumerate(sorted(cat_ids))}
cat_id_to_contiguous = {orig_id : idx for idx, orig_id in enumerate(sorted(cat_ids))}

# for tensors
orig_ids = torch.tensor(list(cat_id_to_contiguous.keys()), dtype=torch.long)
contig_ids = torch.tensor(list(cat_id_to_contiguous.values()), dtype=torch.long)

max_orig_id = orig_ids.max().item()
# 建一个填充–1 的查表张量
mapping = torch.full((max_orig_id + 1,), -1, dtype=torch.long)
# 把对应位置填上连续索引
mapping[orig_ids] = contig_ids
mapping = mapping.to(device)

cfg['mapping'] = mapping


## end
# 提取名称列表
cat_names = {cat['id'] : cat['name'] for cat in cats}   # e.g. ['person','bicycle','car',…] :contentReference[oaicite:1]{index=1}

conf_thresh = float(cfg['conf_thresh'])

unnormalize = torchvision.transforms.Normalize(
    mean=[-m/s for m, s in zip(mean, std)],
    std=[1/s for s in std]
)

def save_tensor_box(image, targets, label, outputs, pred_cl, pred_conf, pic_num):
    # 反归一化
    img_denorm = unnormalize(image)
    img_denorm = img_denorm.clamp(0,1) * 255
    img_uint8 = img_denorm.to(torch.uint8)

    # 获取标注类别
    label = label.long()
    labels_list = label.tolist()         # [1,3,17,1]
    names_list_truth  = [cat_names[label_] for label_ in labels_list ]
    
    # 获取
    cls_conf, cls_idx = pred_cl.max(dim=1)
    final_scores = pred_conf * cls_conf
    mask = final_scores >= conf_thresh
    outputs = outputs[mask]
    scores  = final_scores[mask]
    pred_classes = cls_idx[mask]

    #mask = pred_conf >= conf_thresh
    #mask = mask.tolist()
    #pred_cl = pred_cl.max(dim=1).indices
    #pred_cl = pred_cl.long()
    #outputs = outputs[mask]
    #scores = pred_conf[mask]

    targets = box_convert(targets, in_fmt='xywh', out_fmt='xyxy')
    outputs = box_convert(outputs, in_fmt='cxcywh', out_fmt='xyxy')



    keep = batched_nms(outputs, scores, pred_classes, iou_threshold=0.45)



    #keep = nms(boxes=outputs, scores=scores, iou_threshold=0.45)
    outputs = outputs[keep]

    #names_list_pred = [names_list_pred[i] for i in keep.tolist()]  # List[str]
    pred_classes = pred_classes[keep]
    labels_list = (pred_classes).tolist()        # [1,3,17,1]
    names_list_pred = [cat_names[cat_id_to_incontiguous[label_]] for label_ in labels_list ]
    #scores = scores[keep]
    #names_list_pred = [names_list_pred[i] for i in keep.tolist()]

    img_truth = draw_bounding_boxes(img_uint8.to(torch.uint8), targets, labels=names_list_truth, colors="red")
    img_pred = draw_bounding_boxes(img_uint8.to(torch.uint8), outputs, labels=names_list_pred, colors="green")

    pil_gt = to_pil_image(img_truth)
    pil_pred = to_pil_image(img_pred)
    pil_gt.save(f"results/pictures/{pic_num}_gt_visualization.png")
    pil_pred.save(f"results/pictures/{pic_num}_pred_visualization.png")












'''
    keep = []

    for cls in set(names_list_pred):                          # 1. 遍历每个类别名
        # 2. 收集该类别在原列表中的索引
        idxs = [i for i, c in enumerate(names_list_pred) if c == cls]
        if not idxs:
            continue
        # 3. 转成 LongTensor，用来从 boxes/scores 中切片
        idxs_t = torch.tensor(idxs, device=outputs.device)
        cls_boxes  = outputs[idxs_t]                # Tensor[M,4]
        cls_scores = scores[idxs_t]                    # Tensor[M]
        # 4. 对单个类别做 NMS，得到保留框在 cls_boxes 中的相对索引
        keep_idxs = nms(cls_boxes, cls_scores, iou_threshold=0.45)
        # 5. 映射回全局索引
        keep.extend([idxs[i] for i in keep_idxs.tolist()])

    # 6. 最终索引转 Tensor
    keep = torch.tensor(keep, device=outputs.device)
'''