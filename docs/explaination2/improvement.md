# 提升
## ap计算
通过修改数据集代码，在标注的开头增加一个固定的image_id，然后在验证函数增加对AP的处理，调用COCO API，然后通过调节freq参数调节评估的频率，因为这个很耗时。然后我因此增加了两个参数anns_ap,coco_result。

## parser
