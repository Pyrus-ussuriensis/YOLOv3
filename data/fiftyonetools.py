#imgs_train = cfg['imgs_train']
import fiftyone as fo
import fiftyone.types as fot
from src.utils.cfg import cfg

dataset = fo.Dataset.from_dir(
    dataset_dir="data/",
    dataset_type=fot.COCODetectionDataset,
    data_path="coco/images/train2017",                          # 相对 dataset_dir
    labels_path="coco/annotations/instances_train2017.json",     # 相对 dataset_dir
    name="coco_train2017",
    label_field="ground_truth_detections",
)
session = fo.launch_app(dataset)


'''
# 3. 添加预测标签字段
dataset.add_predictions(
    "predictions.json",
    label_field="predictions",
    parser=fot.COCODetectionDataset
)

# 4. 评估检测结果：默认 mAP@[.5:.95]
results = dataset.evaluate_detections(
    "predictions",
    gt_field="ground_truth",
    eval_key="coco_eval"
)

# 5. 打印评估指标
print(results.metrics())
# 支持 AR、AP 在不同阈值和对象尺寸上的分解

'''