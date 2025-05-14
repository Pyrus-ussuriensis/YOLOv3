4. 参数说明
5. 最终解释

# 参数解释
```yaml
# general config
device: auto   # 可选值：auto、cuda、cpu
experiment: 9 # 实验标号，影响Tensorboard和log
load: 8 # 加载的权重文件号
save: 8 # 保存的权重文件号
batch_size: 16
freq: 10 # 显示可视化结果的epoch间隔数
train_subset_len: 20000 # train模式下划分子集大小                    
val_subset_len: 1000                    
subset_len: 1 # 给出subset的大小
mode: 'full_train' # test train full_train，分别用sub，subtrain，train，第二个是根据前面划分的大小

# paths
weights_path: 'weights/' # 权重路径
anns_train: 'data/coco/annotations/annotations_trainval2017/instances_train2017.json' # 标注路径
anns_val: 'data/coco/annotations/annotations_trainval2017/instances_val2017.json'
anns_sub: 'data/sub/instances_sub20.json'
imgs_train: 'data/coco/images/train2017/' # 图片路径
imgs_val: 'data/coco/images/val2017/'
cache_train: 'data/cache/train.pkl' # 缓存路径
cache_val: 'data/cache/val.pkl'
cache_sub: 'data/cache/sub.pkl'

IMAGENET_ROOT: 'data/imagenet' # imagenet路径
imagenet_train: 'data/imagenet/train' # COCO数据集路径
imagenet_val: 'data/imagenet/val'

# constant values
mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]



# hypoparameters
lr: 1e-4            # 学习率  
momentum: 0.9          # 动量 原本用于SGD
weight_decay: 0.0005  # 权重衰减
epochs: 600          # 迭代次数  
img_dim: 320
pic_size: 320
num_classes: 80
ignore_thresh: '0.5' # 阈值
truth_thresh: '1'
conf_thresh: '0.5'

step_size: 30
gamma: 0.1

```

# 最终解释
由于我开始用train模式，我以为能在3天后结束训练，但用完整则要半个月，所以我放弃了。