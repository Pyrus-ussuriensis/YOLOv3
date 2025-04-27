import torch
from src.utils.cfg import cfg
from src.utils.log import logger

experiment = cfg['experiment']
weights_path = cfg['weights_path']

# 加载和保存权重文件
def load_model(model, i=experiment):
    model.load_state_dict(torch.load(weights_path+f'RTST_{i}.pth'))
    logger.info("successfully load model")
def save_model(model, i=experiment):
    torch.save(model.state_dict(),weights_path+f'RTST_{i}.pth')
    logger.info("successfully save model")
