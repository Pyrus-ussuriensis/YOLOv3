# learning notes
## datasets
这次是用COCO数据集，读取和之前不同，它分为图片和标注两个部分。
在本次任务中，我们在训练和验证时使用 **annotations_trainval2017** 这个文件下的instance标注，使用COCO API读取，
安装命令
```bash
conda install -c conda-forge pycocotools
```
测试集读取 **image_info_test2017** ，它不包含任何标注，仅提供测试集中图片基本信息，如编号，名称，长，宽，但是我们仍可以利用fiftyone可视化。

### 利用COCO API使用数据集
```python
from torchvision.datasets import CocoDetection
dataset = CocoDetection(
    root="images/train2017",
    annFile="annotations/instances_train2017.json",
    transforms=my_transforms
)
```
通过CocoDetection处理得到的数据集是Dataset的子类，我们可以直接使用。然后每个对象的格式是(PIL.Image, List[dict])，其中每个 dict 包含 bbox、category_id、segmentation 等，bbox的格式是一个列表包含一个点的横纵坐标和框的长宽，category_id给一个类别值，segmentation给分割的多边形的所有点，横纵坐标依次在一个一维列表中。
transforms会处理标注和图片，所以一般这么做。注意 **裁切时要同步变换标注** ，之后要学习collate_fn，padding等参数的使用。
```python
import torchvision.transforms as T

def train_transform(image, target):
    # 1) 随机缩放与裁剪
    # 2) 将 PIL 转为 Tensor
    image = T.ToTensor()(image)
    # 3) 归一化
    image = T.Normalize(mean, std)(image)
    # 4) 根据缩放裁剪更新 target["bbox"]
    return image, target

dataset = CocoDetection(root, annFile, transforms=train_transform)
```
如果不是实现自定义类应该不需要使用pycocotools.COCO，下面是一个两个例子。
```bash
from pycocotools.coco import COCO

# 初始化真实注释 COCO 对象
cocoGt = COCO("annotations/instances_train2017.json")

# 获取所有 person 类别的图像 IDs
person_cat = cocoGt.getCatIds(catNms=["person"])
img_ids    = cocoGt.getImgIds(catIds=person_cat)

# 加载前 5 张图像的元数据与注释
imgs = cocoGt.loadImgs(img_ids[:5])
for img in imgs:
    anns = cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=[img["id"]], catIds=person_cat))
    masks = [cocoGt.annToMask(ann) for ann in anns]  # 二值掩码列表
```

```python
from pycocotools.coco import COCO
import os
from PIL import Image
import matplotlib.pyplot as plt

# 1. 指定 COCO Detection 标注文件
ann_file = "coco/annotations/instances_train2017.json"
coco     = COCO(ann_file)

# 2. 列出所有类别名称
cats = coco.loadCats(coco.getCatIds())
print("COCO Categories:", [c["name"] for c in cats])

# 3. 获取包含类别 'person' 的图像 IDs，并加载前 3 张图像
person_id = coco.getCatIds(catNms=["person"])[0]
img_ids   = coco.getImgIds(catIds=[person_id])[:3]
imgs      = coco.loadImgs(img_ids)

for img_info in imgs:
    # 4. 加载并显示图像
    img_path = os.path.join("coco/images/train2017", img_info["file_name"])
    img      = Image.open(img_path)
    plt.imshow(img); plt.axis("off")
    
    # 5. 获取并可视化该图像的所有检测框
    ann_ids = coco.getAnnIds(imgIds=[img_info["id"]], catIds=[person_id])
    anns    = coco.loadAnns(ann_ids)
    coco.showAnns(anns)  # 在当前图像上绘制边界框
    plt.show()
```
COCO API提供了处理我们得到的结果和标注的处理方式，以及结果的评估。
```python
from pycocotools.cocoeval import COCOeval
cocoGt = coco                          # 真实注释
cocoDt = cocoGt.loadRes("preds.json")  # 预测结果
eval  = COCOeval(cocoGt, cocoDt, iouType="bbox")
eval.evaluate(); eval.accumulate(); eval.summarize()
```
evaluate()会在不同阈值下计算给出框是否为真，按照置信度排列，存入self.evalImgs。
accumulate()评估结果整理到Precision-Recall 曲线和 Recall 曲线，存入self.eval['precision'] 与 self.eval['recall']
summarize()，给出各个AP，AR指标。

由于标注中不同图片的标注数量不同，所以标注不能拼接，所以我们要修改DataLoader的collate。具体而言collate_fn 接收 batch，其类型为 List[ Tuple[image, target] ]（长度为 batch_size），然后我们将图片编成一个张量，然后标注维持为列表，但都转化为元组。
```python
def detection_collate_fn(batch):
    # batch: list of (image_tensor, target_dict)
    return tuple(zip(*batch))
```
与其训练时在整合成一个张量，可以在这里整合。
```python
def detection_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, targets
```
利用Albumentations实现对图片的裁切缩放，它会自动改相应的标注，它基于Numpy，在CPU上使用。
### fiftyone