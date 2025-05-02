```python
from torchvision import transforms
from torchvision.utils import save_image
from src.utils.log import logger
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from pycocotools.coco import COCO
from src.utils.cfg import cfg
from torchvision.ops import box_convert

# 指定 COCO 注释文件路径
anns_train = cfg['anns_train']
mean = cfg['mean']
std = cfg['std']

# 初始化 COCO 对象（会加载并索引 JSON）
coco = COCO(anns_train)

# 获取所有类别 ID，然后加载对应的类别信息
cat_ids = coco.getCatIds()                  # e.g. [1, 2, 3, …, 90] :contentReference[oaicite:0]{index=0}
cats    = coco.loadCats(cat_ids)            # 列表，每项如 {'id':1,'name':'person',…}
cat_id_to_contiguous = {idx : orig_id for idx, orig_id in enumerate(sorted(cat_ids))}
# 提取名称列表
cat_names = {cat['id'] : cat['name'] for cat in cats}   # e.g. ['person','bicycle','car',…] :contentReference[oaicite:1]{index=1}

conf_thresh = float(cfg['conf_thresh'])

unnormalize = torchvision.transforms.Normalize(
    mean=[-m/s for m, s in zip(mean, std)],
    std=[1/s for s in std]
)

```
获取参数，得到标注类别和实际类别（数字到字符串）的对应字典。**cat_names**
由于COCO数据集类别不是连续的，空了一些类别数字，建立连续数字和实际列表的对应关系。**cat_id_to_contiguous**
反归一化处理。

```python
def save_tensor_box(image, targets, label, outputs, pred_cl, pred_conf, pic_num):
  img_denorm = unnormalize(image)
  img_denorm = img_denorm.clamp(0,1) * 255
  img_uint8 = img_denorm.to(torch.uint8)

  label = label.long()
  labels_list = label.tolist()         # [1,3,17,1]
  names_list_truth  = [cat_names[label_] for label_ in labels_list ]
  
  mask = pred_conf >= conf_thresh
  mask = mask.tolist()
  pred_cl = pred_cl.max(dim=1).values
  pred_cl = pred_cl.long()
  labels_list = (pred_cl[mask]).tolist()        # [1,3,17,1]
  names_list_pred = [cat_names[cat_id_to_contiguous[label_]] for label_ in labels_list ]
  outputs = outputs[mask]

  targets = box_convert(targets, in_fmt='xywh', out_fmt='xyxy')
  outputs = box_convert(outputs, in_fmt='xywh', out_fmt='xyxy')
  img_truth = draw_bounding_boxes(img_uint8.to(torch.uint8), targets, labels=names_list_truth, colors="red")
  img_pred = draw_bounding_boxes(img_uint8.to(torch.uint8), outputs, labels=names_list_pred, colors="green")

  pil_gt = to_pil_image(img_truth)
  pil_pred = to_pil_image(img_pred)
  pil_gt.save(f"results/pictures/{pic_num}_gt_visualization.png")
  pil_pred.save(f"results/pictures/{pic_num}_pred_visualization.png")


```
首先对图片反归一化。
然后根据列表一维张量转化成类别列表，包括将浮点数转化为整数，转化为列表，然后将数字转化为实际类别。
后面预测我们仅显示置信度达到一定数值的，预测张量要获取最大值为预测类别张量，然后转化为整数，用掩码处理，转化为列表，转化为实际类别。
这是因为画框的函数需要标注的形式是[N,4]张量，类别是字符串列表。但要求的框是四个角的坐标而不是长宽，所以要先转化。
最后转化为PIL.Image，显示。