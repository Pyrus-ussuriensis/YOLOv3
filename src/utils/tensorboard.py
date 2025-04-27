from torch.utils.tensorboard import SummaryWriter
from utils.cfg import cfg
experiment = cfg["experiment"]

# 建立Tensorboard的写对象
writer = SummaryWriter(log_dir='tensorboard/'+f'experiment{experiment}')