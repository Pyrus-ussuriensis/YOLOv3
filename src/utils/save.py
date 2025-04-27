from torchvision import transforms
from torchvision.utils import save_image
from src.utils.log import logger

# 保存单张图片，即Batch的第一张图片
def save_pic(target, i, mode):
  denormalization = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
  img = target.clone().squeeze()
  img = denormalization(img).clamp(0, 1)
#设定保存的路径和文件名
  if mode == 'origin':
    save_image(img, f'results/store/origin_{i}.png')
  elif mode == 'generate':
    save_image(img, f'results/store/generate_{i}.png')
  else:
    logger.error("wrong image save mode!!!")
    

# 保存Batchsize个图片
def save_pics(target, i, mode):
  n, _, _, _ = target.shape
  for j in range(n):
    save_pic(target[j:j+1], i+j, mode)
  return n

