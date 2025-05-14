2. backbone和分类头，检测头的搭建
   1. 架构文件的读取
   2. backbone的实际获取

## backbone和分类头，检测头的搭建
### 架构文件的读取
backbone的架构可以从论文中读取，但是完整的架构，包括检测头需要在如GitHub上找到。我找到的是一个[yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)，其我们可以整理这个文件内容得到一层层的架构设计然后写出实际架构。
models/yolov3.cfg
models/yolov3_cfg.py
### backbone的实际获取及模型
余下的模型
backbone_g中的是我从[这里](https://github.com/developer0hye/PyTorch-Darknet53)得到的backbone的文件及其权重，因为backbone需要在我的笔记本上训练一个月所以我放弃了，get_backbone读取权重得到实际的backbone，对于原本的darknet53我直接改了forward函数，取出了前两层的输出，去掉了分类头，然后加载权重时strict=False。
backbone是我写的backbone，Classifier是我写的训练backbone用的分类头，Darknet53Classifier是对前两者的整合。detectionheads是检测头，YOLOv3是对backbone和检测头的整合。