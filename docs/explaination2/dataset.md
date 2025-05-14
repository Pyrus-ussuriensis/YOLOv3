1. 数据集的准备
   1. 数据集准备
   2. 小子集
   3. ImageNet准备
      1. 验证集整理
      2. ImageNet准备


## 训练集
主要的工作在getitem中实现使用orjson工具从json文件中读取信息，为了加速可以利用pickle做缓存，做一次之后仅读取缓存，然后得到标注，利用COCO API，然后获取图片，做标注和图片的预处理，然后在DataLoader中实现对图片做张量整合，以batch为单位，而对标注由于每个不同，放到列表中。
data/coco_dataset.py

## 小子集
小子集，我需要利用COCO API随机读取编号类别取出相应的图片和标注然后组成新的json作为用于过拟合的小子集的标注，图片路径写训练集的就可以了，然后COCO API会根据编号一一获取。由于缓存的存在，如果要换标注，需要删除原有缓存。
data/sub/subset.py

## ImageNet准备
### 验证集准备
验证集原始文件中图片没有按照类别分类，所以我们需要按照类别放到一个个文件夹中。因为torchvision.datasets的ImageFolder, ImageNet类或者需要类别号文件夹。
具体我们需要利用kit读取相关信息，移动图片。
data/tools/val_classes.py

### ImageNet准备
仅需要利用ImageFolder或者ImageNet，后者接受压缩包然后解压，前者接受文件夹。然后我们将路径给它，给其预处理的transform。
data/imagenet_dataset.py




