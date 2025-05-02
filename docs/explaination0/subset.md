```python
from pycocotools.coco import COCO
import json, random
from src.utils.cfg import cfg

subset_len = cfg['subset_len']
anns_train = cfg['anns_train']
```
获取参数，COCO API

```python
# 加载原始注释
coco = COCO(anns_train)
# 随机采样图片ID
img_ids = random.sample(coco.getImgIds(), subset_len)
# 加载对应 images 与 annotations
images = coco.loadImgs(img_ids)
ann_ids = coco.getAnnIds(imgIds=img_ids)
annotations = coco.loadAnns(ann_ids)
categories = coco.loadCats(coco.getCatIds())

# 输出新 JSON
sub = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}
with open(f'data/instances_sub{subset_len}.json', 'w') as f:
    json.dump(sub, f)
```
读取标注，随机选一定数量，由参数给出，然后编成标注，之后根据标注在训练集中读取，组成测试集。