```yaml
# general config
device: auto   # 可选值：auto、cuda、cpu
experiment: 0  # 实验号
load: 0 # 加载和保存权重文件号
save: 0
batch_size: 16
freq: 50 # 训练时每隔多少步可视化结果
train_subset_len: 20000  # 划分子集用于训练验证测试参数                    
val_subset_len: 1000                    
subset_len: 20
mode: 'train' # 模式设置

# paths
weights_path: 'weights/' # 权重文件路径
anns_train: 'data/coco/annotations/annotations_trainval2017/instances_train2017.json' # 训练验证测试标注路径
anns_val: 'data/coco/annotations/annotations_trainval2017/instances_val2017.json' 
anns_sub: 'data/instances_sub20.json'
imgs_train: 'data/coco/images/train2017/' # 训练验证图片路径
imgs_val: 'data/coco/images/val2017/' 
cache_train: 'data/cache/train.pkl' # 读取标注后缓存路径
cache_val: 'data/cache/val.pkl'
cache_sub: 'data/cache/sub.pkl'

# constant values
mean: [0.485, 0.456, 0.406] 
std: [0.229, 0.224, 0.225]



# hypoparameters
lr: 1e-3            # 学习率  
epochs: 20          # 迭代次数  
pic_size: 320
img_dim: 320
num_classes: 80 # 类别数
ignore_thresh: '0.7' # IoU和置信度划分的依据
truth_thresh: '1'
conf_thresh: '0.5'

# visualization config #过去项目参数，暂时未修改
cv_mode: 'video'
video_path: 'data/video/4.mp4'
output_path: 'results/video/output_7.mp4'
weight_path: 'weights/rtst_udnie.pth'
```