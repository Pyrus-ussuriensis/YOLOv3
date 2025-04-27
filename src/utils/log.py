import logging
import yaml
from src.utils.cfg import cfg
from logging.handlers import RotatingFileHandler
from src.utils.tensorboard import writer
from pprint import pformat

experiment = cfg['experiment']

# 建立日志记录对象
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# 文件最大 10MB，保留 5 个备份
handler = RotatingFileHandler(f'logs/experiment{experiment}.log', maxBytes=10*1024*1024, backupCount=5)
logger.addHandler(handler)
logger.addHandler(logging.StreamHandler())

# 记录参数的配置
logger.info("current configuration\n%s", pformat(cfg))
logger.info('start training')




# 训练和验证时日志和Tensorboard的统一记录
def log_info(epoch, loss, mode, place):
    info = f"mode: {mode}\n{place}: {epoch}\nloss: {loss}\n\n"
    #print(info)
    logger.info(info)
    if mode == "train":
        if place == 'epoch':
            writer.add_scalar('train/loss', loss, epoch)
    elif mode == "val":
        if place == 'epoch':
            writer.add_scalar('val/loss', loss, epoch)
    else:
        error_info = "writer mode error!!!"
        #print(error_info)
        logger.error(error_info)
    writer.flush()
        


