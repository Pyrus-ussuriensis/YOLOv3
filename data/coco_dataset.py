from pycocotools.coco import COCO
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader, random_split
from src.utils.cfg import cfg
import pickle, os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torchvision.datasets.vision import VisionDataset
import orjson


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

cache_train = cfg['cache_train']
cache_val = cfg['cache_val']
cache_sub = cfg['cache_sub']
device = cfg['device']

# 预定义管线，格式为 COCO [x,y,w,h]
alb_pipeline = A.Compose(
    [
        A.RandomResizedCrop(size=(pic_size, pic_size), scale=(0.8,1.0), p=1.0),
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

def transforms_coco(image, target, img_id):
    # PIL.Image → NumPy
    image_np = np.array(image)
    # 将 bbox 及 labels 列表从 target 取出
    #image_id = [ann['image_id'] for ann in target] #[ann['image_id'] for ann in target]
    bboxes = [ann['bbox'] for ann in target]
    labels = [ann['category_id'] for ann in target]
    # 执行变换
    augmented = alb_pipeline(image=image_np, bboxes=bboxes, labels=labels, )
    # 更新
    image = augmented['image']

    # 4. 从 augmented 中取回变换后的框和标签
    new_bboxes = augmented['bboxes']
    new_labels = augmented['labels']

    # 5. 重建 target 列表，保持与原始格式一致
    new_target = [ {'image_id' : img_id} ] + [
        {'bbox': bbox, 'category_id': label}
        for bbox, label in zip(new_bboxes, new_labels)
    ]

    return image, new_target

def detection_collate_fn(batch): # 处理一个batch
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    targets = [*targets]
    for i, target in enumerate(targets):
        targets[i] = [{'image_id': torch.tensor(target[0]['image_id'], dtype=torch.float32, device=device)}] + [
                {'bbox': torch.tensor(t['bbox'], dtype=torch.float32, device=device), 
                'category_id': torch.tensor(t['category_id'], dtype=torch.float32, device=device)}
                for t in target[1:]
            ]
    return images, targets

def load_coco_cached(ann_path: str, cache_path: str):
    if os.path.exists(cache_path):
        # 直接加载缓存
        with open(cache_path, 'rb') as f:
            coco = pickle.load(f)
    else:
        with open(ann_path, 'rb') as f:
            data = orjson.loads(f.read())
        coco = COCO()
        coco.dataset = data
        coco.createIndex()

        # 第一次解析 JSON 并构建索引（慢 ~20s）
        #coco = COCO(ann_path)
        # 序列化 COCO 对象，含 dataset, anns, imgs, imgToAnns 等
        with open(cache_path, 'wb') as f:
            pickle.dump(coco, f, protocol=pickle.HIGHEST_PROTOCOL)
    return coco

class CachedCocoDetection(VisionDataset):
    def __init__(self, root, annFile, cache_file, transforms=None):
        """
        root: 图片目录
        annFile: COCO JSON 标注路径
        cache_file: pickle 缓存路径
        transforms: Albumentations 变换函数（返回 image, target_list）
        """
        super().__init__(root, transforms=transforms)
        # 加载或缓存 COCO 对象
        self.coco = load_coco_cached(annFile, cache_file)
        # 图像 id 列表
        self.ids  = list(self.coco.imgs.keys())
        self.root = root
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        # 1) 读图
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        img  = Image.open(os.path.join(self.root, path)).convert('RGB')

        # 2) 取标注（list of dict）
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns    = self.coco.loadAnns(ann_ids)

        # 3) 调用原有 transforms_coco 处理
        if self.transforms:
            img, anns = self.transforms(img, anns, img_id)

        return img, anns


TrainData = CachedCocoDetection(
    root=imgs_train,
    annFile=anns_train,
    cache_file=cache_train,
    transforms=transforms_coco
)

ValData = CachedCocoDetection(
    root=imgs_val,
    annFile=anns_val,
    cache_file=cache_val,
    transforms=transforms_coco
)

SubData = CachedCocoDetection(
    root=imgs_train,
    annFile=anns_sub,
    cache_file=cache_sub,
    transforms=transforms_coco
)

# 根据模式确定要加载的数据集
if mode == 'train':
    TrainData, _ = random_split(TrainData, [train_subset_len, len(TrainData)-train_subset_len])
    ValData, _ = random_split(ValData, [val_subset_len, len(ValData)-val_subset_len])
elif mode == 'test':
    batch_size = 1
elif mode == 'full_train':
    pass

TrainLoader = DataLoader(dataset=TrainData, batch_size = batch_size, collate_fn=detection_collate_fn)
ValLoader = DataLoader(dataset=ValData, batch_size = batch_size, collate_fn=detection_collate_fn)
SubLoader = DataLoader(dataset=SubData, batch_size=batch_size, collate_fn=detection_collate_fn)

if __name__ == '__main__':
    for step, batch in enumerate(TrainLoader):
        images, targets = batch
        device = torch.device('cuda')
        '''
        targets = targets[0]
        targets = [
            {'bbox': torch.tensor(t['bbox'], dtype=torch.float32, device=device), 'category_id': torch.tensor(t['category_id'], dtype=torch.float32, device=device)}
            for t in targets
        ]
        '''

        print(step)
        print(images, targets)