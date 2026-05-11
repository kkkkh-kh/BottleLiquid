# BottleLiquid

用于透明或半透明塑料饮料瓶 ROI 图像的二分类：判断瓶内“无液体 / 有液体”。

项目默认使用 ImageNet 预训练 ResNet18，并冻结 backbone，只训练最后的全连接分类层。所有脚本都支持命令行参数，路径不写死。

## 项目结构

```text
BottleLiquid/
├── data/
│   ├── images/
│   ├── roi_images/
│   ├── annotations/
│   │   └── labels.csv
│   └── splits/
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
├── src/
│   ├── crop_roi.py
│   ├── split_dataset.py
│   ├── dataset.py
│   ├── model.py
│   ├── train_binary.py
│   ├── evaluate_binary.py
│   └── predict_one.py
├── outputs/
│   └── binary_resnet18/
├── requirements.txt
└── README.md
```

## 1. 安装依赖

推荐新建独立环境：

```bash
conda create -n bottle-liquid-yolo python=3.10 pip -y
conda activate bottle-liquid-yolo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

也可以使用 `environment.yml`：

```bash
conda env create -f environment.yml
conda activate bottle-liquid-yolo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

如果你已经在合适的环境中，也可以直接安装：

```bash
pip install -r requirements.txt
```

如需 GPU 训练，请根据你的 CUDA 版本安装对应的 PyTorch。

## 2. 准备数据

将原始图片放到：

```text
data/images/
```

将标注文件放到：

```text
data/annotations/labels.csv
```

`labels.csv` 格式如下：

```csv
filename,xmin,ymin,xmax,ymax,has_liquid,liquid_level,pose,liquid_type,source
img_0001.jpg,120,80,460,900,1,small,upright,water,self-shot
img_0002.jpg,90,140,520,760,0,none,horizontal,none,self-shot
img_0003.jpg,70,60,510,900,1,large,tilted,tea,self-shot
```

其中 `has_liquid=0` 表示无液体，`has_liquid=1` 表示有液体。

## 3. 划分数据集

如果你的图片已经按文件夹放在项目外部的 `../data` 中，也可以先用预训练 YOLO 自动生成瓶体框标注：

```bash
python src/auto_annotate_yolo.py \
  --input_dir ../data \
  --image_output_dir data/images \
  --label_csv data/annotations/labels.csv \
  --model yolov8n.pt
```

脚本会递归读取 `--input_dir` 下的图片，将图片复制到 `data/images`，并生成符合本项目格式的 `labels.csv`。其中瓶体框来自 YOLO 检测，`has_liquid/liquid_level` 会根据上级文件夹名做弱标签推断，仍建议人工复核。

```bash
python src/split_dataset.py \
  --label_csv data/annotations/labels.csv \
  --output_dir data/splits
```

默认按 `0.7/0.15/0.15` 划分 train/val/test，并尽量按 `has_liquid` 分层抽样。

## 4. 裁剪 ROI

```bash
python src/crop_roi.py \
  --image_dir data/images \
  --label_csv data/annotations/labels.csv \
  --output_dir data/roi_images
```

默认会将标注框四周扩展 `0.05`，并自动防止越界。可通过 `--expand_ratio` 修改。

## 5. 训练模型

```bash
python src/train_binary.py \
  --image_dir data/roi_images \
  --label_csv data/annotations/labels.csv \
  --train_txt data/splits/train.txt \
  --val_txt data/splits/val.txt \
  --output_dir outputs/binary_resnet18 \
  --epochs 30 \
  --batch_size 8 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --freeze_backbone \
  --early_stop_patience 8
```

输出文件：

```text
outputs/binary_resnet18/best_resnet18_binary.pth
outputs/binary_resnet18/train_log.csv
```

如果希望微调整个 ResNet18，可使用：

```bash
python src/train_binary.py ... --no_freeze_backbone
```

## 6. 测试模型

```bash
python src/evaluate_binary.py \
  --image_dir data/roi_images \
  --label_csv data/annotations/labels.csv \
  --test_txt data/splits/test.txt \
  --checkpoint outputs/binary_resnet18/best_resnet18_binary.pth \
  --output_dir outputs/binary_resnet18
```

评估脚本会输出 Accuracy、Precision、Recall、F1-score、Confusion Matrix 和 classification_report。

逐样本预测结果保存到：

```text
outputs/binary_resnet18/test_result.csv
```

## 7. 单张图片预测

```bash
python src/predict_one.py \
  --image_path data/roi_images/img_0001.jpg \
  --checkpoint outputs/binary_resnet18/best_resnet18_binary.pth
```

输出包括：

```text
预测类别：无液体 / 有液体
prob_no_liquid
prob_has_liquid
```

## 8. 四分类液体量识别

如果需要识别 `无 / 少 / 中 / 多` 四类，可使用 Roboflow COCO 标注中的瓶体和液体区域生成四分类标签。

类别编号如下：

```text
0: none   无
1: small  少
2: medium 中
3: large  多
```

默认按 `液体区域面积 / 瓶体区域面积` 生成弱标签：

```text
ratio = 0              -> none
0 < ratio < 0.25      -> small
0.25 <= ratio < 0.60  -> medium
ratio >= 0.60         -> large
```

生成四分类标注：

```bash
python src/prepare_multiclass_from_coco.py \
  --coco_json ../data/train/_annotations.coco.json \
  --source_image_dir ../data/train \
  --output_image_dir data/multiclass/images \
  --label_csv data/annotations/labels_multiclass.csv \
  --debug_csv data/annotations/labels_multiclass_debug.csv
```

划分四分类数据集：

```bash
python src/split_dataset.py \
  --label_csv data/annotations/labels_multiclass.csv \
  --output_dir data/splits_multiclass \
  --stratify_col liquid_class
```

裁剪四分类 ROI：

```bash
python src/crop_roi.py \
  --image_dir data/multiclass/images \
  --label_csv data/annotations/labels_multiclass.csv \
  --output_dir data/multiclass/roi_images
```

训练四分类模型：

```bash
python src/train_multiclass.py \
  --image_dir data/multiclass/roi_images \
  --label_csv data/annotations/labels_multiclass.csv \
  --train_txt data/splits_multiclass/train.txt \
  --val_txt data/splits_multiclass/val.txt \
  --output_dir outputs/multiclass_resnet18 \
  --epochs 30 \
  --batch_size 8 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --freeze_backbone \
  --early_stop_patience 8
```

测试四分类模型：

```bash
python src/evaluate_multiclass.py \
  --image_dir data/multiclass/roi_images \
  --label_csv data/annotations/labels_multiclass.csv \
  --test_txt data/splits_multiclass/test.txt \
  --checkpoint outputs/multiclass_resnet18/best_resnet18_multiclass.pth \
  --output_dir outputs/multiclass_resnet18
```

单张 ROI 四分类预测：

```bash
python src/predict_one_multiclass.py \
  --image_path data/multiclass/roi_images/your_image.jpg \
  --checkpoint outputs/multiclass_resnet18/best_resnet18_multiclass.pth
```

### 整合弱标签数据扩充四分类

除了 COCO 标注数据，也可以把 `Full Water level / Half water level / Overflowing` 文件夹经 YOLO 得到的弱标签数据整合进四分类训练集。该方式会增加数据量，但标签噪声更大，建议和 COCO-only 结果对比。

```bash
python src/combine_multiclass_datasets.py \
  --coco_label_csv data/annotations/labels_multiclass.csv \
  --coco_image_dir data/multiclass/images \
  --weak_label_csv data/annotations/labels.csv \
  --weak_image_dir data/images \
  --output_image_dir data/combined_multiclass/images \
  --output_label_csv data/annotations/labels_multiclass_combined.csv
```

```bash
python src/split_dataset.py \
  --label_csv data/annotations/labels_multiclass_combined.csv \
  --output_dir data/splits_multiclass_combined \
  --stratify_col liquid_class
```

```bash
python src/crop_roi.py \
  --image_dir data/combined_multiclass/images \
  --label_csv data/annotations/labels_multiclass_combined.csv \
  --output_dir data/combined_multiclass/roi_images
```

训练 combined 四分类模型：

```bash
python src/train_multiclass.py \
  --image_dir data/combined_multiclass/roi_images \
  --label_csv data/annotations/labels_multiclass_combined.csv \
  --train_txt data/splits_multiclass_combined/train.txt \
  --val_txt data/splits_multiclass_combined/val.txt \
  --output_dir outputs/multiclass_resnet18_combined \
  --epochs 30 \
  --batch_size 8 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --freeze_backbone \
  --early_stop_patience 8
```

### 加入人工确认的 newtrashy 空瓶样本

如果人工审查认为 `newtrashy` 中的瓶体实例可作为空瓶样本，可随机抽取一部分加入 `none` 类。建议先少量加入，例如 150 个 ROI，避免空瓶域过强导致模型偏移。

```bash
python src/add_newtrashy_none_samples.py \
  --base_label_csv data/annotations/labels_multiclass_combined.csv \
  --base_roi_dir data/combined_multiclass/roi_images \
  --newtrashy_dir ../newtrashy \
  --output_label_csv data/annotations/labels_multiclass_combined_none_aug.csv \
  --output_roi_dir data/combined_multiclass_none_aug/roi_images \
  --num_samples 150 \
  --seed 42
```

```bash
python src/split_dataset.py \
  --label_csv data/annotations/labels_multiclass_combined_none_aug.csv \
  --output_dir data/splits_multiclass_combined_none_aug \
  --stratify_col liquid_class
```

训练 none-aug 四分类模型：

```bash
python src/train_multiclass.py \
  --image_dir data/combined_multiclass_none_aug/roi_images \
  --label_csv data/annotations/labels_multiclass_combined_none_aug.csv \
  --train_txt data/splits_multiclass_combined_none_aug/train.txt \
  --val_txt data/splits_multiclass_combined_none_aug/val.txt \
  --output_dir outputs/multiclass_resnet18_combined_none_aug \
  --epochs 50 \
  --batch_size 8 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --freeze_backbone \
  --early_stop_patience 8
```

## 说明

- 所有脚本都可以在 CPU 上运行，速度会慢一些。
- 首次构建 ResNet18 会下载 ImageNet 预训练权重，需要联网。
- 如果图片、标注列、split 文件或 checkpoint 缺失，脚本会给出明确报错。
- 本任务只训练二分类标签 `has_liquid`，`liquid_level`、`pose`、`liquid_type` 等字段暂时保留，不参与训练。
