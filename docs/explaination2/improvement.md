# 提升
## ap计算
通过修改数据集代码，在标注的开头增加一个固定的image_id，然后在验证函数增加对AP的处理，调用COCO API，然后通过调节freq参数调节评估的频率，因为这个很耗时。然后我因此增加了两个参数anns_ap,coco_result。

## parser
今天在train.py函数中增加了对于命令行参数的支持，主要我支持如下参数的设置。
```yaml
experiment: 9 # 实验标号，影响Tensorboard和log
load: 8 # 加载的权重文件号
save: 8 # 保存的权重文件号
batch_size: 16
mode: 'full_train' # test train full_train，分别用sub，subtrain，train，第二个是根据前面划分的大小

lr: 1e-4            # 学习率  
momentum: 0.9          # 动量 原本用于SGD
weight_decay: 0.0005  # 权重衰减
epochs: 600          # 迭代次数  
step_size: 30
gamma: 0.1
```
分别可以通过
* --experiment -e
* --load -l
* --save -s
* --batchsize -b
* --mode -m
* --lr
* --momentum
* --weight -w
* --epoch 
* --step 
* --gamma -g
由于导入cfg会有一次日志记录，所以，真正根据命令行参数修改得到的最终参数是第二次打印所有参数的显示结果。
