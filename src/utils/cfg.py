import yaml
import torch
from PIL import Image
from torchvision import transforms

# 在这里统一读取参数，放到一个字典中，如果要读取参数，导入这个文件读取
with open("configs/yolo.yaml","r",encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

# 一些其他参数可以在这里统一加入
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg['device'] = device
mean = cfg['mean']
std = cfg['std']

transform_pic_o = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)

        ])


anchors = [ (10,13),(16,30),(33,23),(30,61),(62,45),(59,119),(116,90),(156,198),(373,326) ]
cfg['anchors'] = anchors


# 根据模式修改部分参数
mode = cfg['mode']
if mode == 'test':
    cfg['epochs'] = 2000          # 迭代次数  
    cfg['batch_size'] = 1
    cfg['freq'] = 5