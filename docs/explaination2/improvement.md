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

## 学习
这里我主要学习这个[项目](https://github.com/eriklindernoren/PyTorch-YOLOv3)

### poetry
* 功能
这个工具的用途是可以管理python环境，同时方便的打包发布供别人复现
* 与conda的关系
conda不仅限于python环境，而这个仅限于python，一般主流深度学习就使用conda，不推荐二者结合，容易弄乱
* 项目中创建
我们在新项目中使用poetry new <name>创建一个新项目，如果不是在conda中则会在.venv横纵创建新的virtualenv，如果在则直接使用conda的环境不会重新创建也不会尝试管理，如果我们要用，必须一个个添加相关依赖的名称和版本。
对于一个已经完成的项目，用poetry init，交互式生成对依赖的详细描述。
* 项目中管理依赖
  * poetry add 添加依赖
  * poetry remove 去除依赖
  * poetry update 更新依赖
  * poetry lock 更新依赖锁文件
  * poetry show --tree 显示依赖树
* 项目中运行
  * poetry run 不激活虚拟环境，后面跟实际命令如python之类运行
  * poetry shell 激活环境
* 打包发布使用
  * poetry build 生成发布包
  * poetry publish 发布至PyPI
  * poetry install 根据依赖文件创建环境
* 详细说明
通过pyproject.toml poetry.lock两个文件对依赖进行管理，前者记录元数据如作者名称证书等，简单的依赖记录，即仅记录大概的版本范围，后者进行精确的依赖版本记录。
* 其他
我们还能再pyproject.toml中声明命令，在安装后直接使用。

### 设计
这个项目的设计将相关的函数和对象封装到一个类中，而不是像我一样直接将相关的函数放到一个文件中，然后直接实例化，在其他文件中调用。他统一在训练文件中实例化，它的应该更科学合理。

### 模型建立
这里他的做法是读取cfg文件，根据读取到的内容一各个写入列表，表中每个对象包含几个要素，然后其他函数可以根据这些信息一一在nn.Sequential()的基础上添加，然后返回完整的模型架构。