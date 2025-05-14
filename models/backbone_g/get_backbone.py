import torch
from models.backbone_g.model import darknet53
from src.utils.cfg import cfg

num_classes = int(cfg['num_classes'])

# 1) 实例化模型（num_classes 可任意设，加载后将抛弃）
backbone = darknet53(num_classes=1000)

# 2) 加载 checkpoint
ckpt = torch.load('models/backbone_g/model_best.pth.tar', map_location='cpu')
# 部分 checkpoint 可能以 {'state_dict': ...} 存储
state_dict = ckpt.get('state_dict', ckpt)
# 3) 与模型对应加载
backbone.load_state_dict(state_dict, strict=False)

# 假设 backbone.modules() 顺序不变，我们可直接删除最后两层
feature_extractor = torch.nn.Sequential(
    *list(backbone.children())[:-2]  # 去掉 global_avg 和 fc
)

#for param in backbone.parameters():
#    param.requires_grad = False