# BottleLiquid

透明/半透明塑料饮料瓶内液体识别项目。项目支持两个任务：

- 二分类：判断瓶内是否有液体，即 `无液体 / 有液体`
- 四分类：判断瓶内液体量等级，即 `无 / 少 / 中 / 多`

分类模型默认使用 ImageNet 预训练的 ResNet18，并冻结 backbone，只训练最后的全连接分类层。项目也提供独立的 YOLO 瓶体检测组件，用于学习瓶体位置并为分类模型裁剪统一 ROI。

## 1. 项目结构

```text
BottleLiquid/
├── data/
│   ├── model_dataset/
│   │   ├── annotations/
│   │   │   └── labels.csv
│   │   ├── roi_images/
│   │   │   └── *.jpg / *.jpeg / *.png
│   │   └── splits/
│   │       ├── train.txt
│   │       ├── val.txt
│   │       └── test.txt
│   ├── bottle_detection/
│   │   ├── images/train, val, test
│   │   ├── labels/train, val, test
│   │   └── data.yaml
│   └── detector_roi_dataset/
│       ├── annotations/labels.csv
│       ├── roi_images/
│       └── splits/
│
├── src/
│   ├── detection/
│   │   ├── prepare_newtrashy_bottle_dataset.py
│   │   ├── train_bottle_detector.py
│   │   └── crop_roi_with_detector.py
│   ├── dataset.py
│   ├── model.py
│   ├── train_binary.py
│   ├── evaluate_binary.py
│   ├── predict_one.py
│   ├── train_multiclass.py
│   ├── evaluate_multiclass.py
│   └── predict_one_multiclass.py
│
├── outputs/
│   └── bottle_detector/
├── requirements.txt
├── environment.yml
└── README.md
```

当前最终数据集统一放在：

```text
data/model_dataset/
```

训练和测试只需要这一套数据。

## 2. 团队成员如何运行

### 2.1 克隆项目

```bash
git clone <你的 GitHub 仓库地址>
cd BottleLiquid
```

如果仓库外层还有一级目录，例如 `IsThereWater`，则进入：

```bash
cd IsThereWater/BottleLiquid
```

后续所有命令都默认在 `BottleLiquid` 目录下执行。

### 2.2 创建 Python 环境

推荐使用 conda：

```bash
conda create -n bottle-liquid-yolo python=3.10 pip -y
conda activate bottle-liquid-yolo
```

安装 CPU 版 PyTorch：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

安装其他依赖：

```bash
pip install -r requirements.txt
```

如果队友有 NVIDIA GPU，可以根据自己的 CUDA 版本安装对应的 PyTorch GPU 版本。

## 3. 数据和权重如何放置

GitHub 仓库建议只上传代码，不上传图片数据和模型权重。数据和权重建议通过网盘共享。

队友下载数据后，目录应放成这样：

```text
BottleLiquid/
├── data/
│   └── model_dataset/
│       ├── annotations/
│       │   └── labels.csv
│       ├── roi_images/
│       │   └── ...
│       └── splits/
│           ├── train.txt
│           ├── val.txt
│           └── test.txt
│
└── outputs/
    ├── binary_resnet18_model_dataset/
    │   └── best_resnet18_binary.pth
    └── multiclass_resnet18_model_dataset/
        └── best_resnet18_multiclass.pth
```

如果没有权重文件，也可以重新训练生成。

## 4. 数据集说明

最终整理后的数据集位于：

```text
data/model_dataset/
```

总样本数：

```text
1442 张 ROI 图片
```

类别定义：

```text
二分类 has_liquid:
0 = 无液体
1 = 有液体

四分类 liquid_class:
0 = none   无
1 = small  少
2 = medium 中
3 = large  多
```

四分类数量分布：

```text
large:   540
medium:  379
none:    324
small:   199
```

二分类数量分布：

```text
有液体: 1118
无液体:  324
```

数据划分：

```text
train: 1009
val:    216
test:   217
```

`labels.csv` 字段说明：

```text
filename,xmin,ymin,xmax,ymax,has_liquid,liquid_level,liquid_class,pose,liquid_type,source
```

其中：

- `filename`：对应 `roi_images/` 下的 ROI 图片文件名
- `has_liquid`：二分类标签，`0` 无液体，`1` 有液体
- `liquid_level`：四分类文本标签，`none/small/medium/large`
- `liquid_class`：四分类数字标签，`0/1/2/3`
- `source`：样本来源，用于追溯
- `xmin,ymin,xmax,ymax`：保留的框字段，用于兼容和追溯；当前模型直接读取 ROI 图片，不再重新裁剪

## 5. 二分类：判断有无液体

### 5.1 训练

```bash
python src/train_binary.py \
  --image_dir data/model_dataset/roi_images \
  --label_csv data/model_dataset/annotations/labels.csv \
  --train_txt data/model_dataset/splits/train.txt \
  --val_txt data/model_dataset/splits/val.txt \
  --output_dir outputs/binary_resnet18_model_dataset \
  --epochs 50 \
  --batch_size 8 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --freeze_backbone \
  --early_stop_patience 8
```

Windows PowerShell 可以使用：

```powershell
python src/train_binary.py `
  --image_dir data/model_dataset/roi_images `
  --label_csv data/model_dataset/annotations/labels.csv `
  --train_txt data/model_dataset/splits/train.txt `
  --val_txt data/model_dataset/splits/val.txt `
  --output_dir outputs/binary_resnet18_model_dataset `
  --epochs 50 `
  --batch_size 8 `
  --lr 0.001 `
  --weight_decay 0.0001 `
  --freeze_backbone `
  --early_stop_patience 8
```

输出：

```text
outputs/binary_resnet18_model_dataset/best_resnet18_binary.pth
outputs/binary_resnet18_model_dataset/train_log.csv
```

### 5.2 测试

```powershell
python src/evaluate_binary.py `
  --image_dir data/model_dataset/roi_images `
  --label_csv data/model_dataset/annotations/labels.csv `
  --test_txt data/model_dataset/splits/test.txt `
  --checkpoint outputs/binary_resnet18_model_dataset/best_resnet18_binary.pth `
  --output_dir outputs/binary_resnet18_model_dataset
```

输出指标包括：

```text
Accuracy
Precision
Recall
F1-score
Confusion Matrix
classification_report
```

逐样本预测结果保存到：

```text
outputs/binary_resnet18_model_dataset/test_result.csv
```

### 5.3 单张图片预测

```powershell
python src/predict_one.py `
  --image_path data/model_dataset/roi_images/your_image.jpg `
  --checkpoint outputs/binary_resnet18_model_dataset/best_resnet18_binary.pth
```

输出：

```text
预测类别：无液体 / 有液体
prob_no_liquid
prob_has_liquid
```

## 6. 四分类：判断无 / 少 / 中 / 多

### 6.1 训练

```powershell
python src/train_multiclass.py `
  --image_dir data/model_dataset/roi_images `
  --label_csv data/model_dataset/annotations/labels.csv `
  --train_txt data/model_dataset/splits/train.txt `
  --val_txt data/model_dataset/splits/val.txt `
  --output_dir outputs/multiclass_resnet18_model_dataset `
  --epochs 50 `
  --batch_size 8 `
  --lr 0.001 `
  --weight_decay 0.0001 `
  --freeze_backbone `
  --early_stop_patience 8
```

输出：

```text
outputs/multiclass_resnet18_model_dataset/best_resnet18_multiclass.pth
outputs/multiclass_resnet18_model_dataset/train_log_multiclass.csv
```

### 6.2 测试

```powershell
python src/evaluate_multiclass.py `
  --image_dir data/model_dataset/roi_images `
  --label_csv data/model_dataset/annotations/labels.csv `
  --test_txt data/model_dataset/splits/test.txt `
  --checkpoint outputs/multiclass_resnet18_model_dataset/best_resnet18_multiclass.pth `
  --output_dir outputs/multiclass_resnet18_model_dataset
```

输出指标包括：

```text
Accuracy
Macro Precision
Macro Recall
Macro F1-score
Confusion Matrix
classification_report
```

逐样本预测结果保存到：

```text
outputs/multiclass_resnet18_model_dataset/test_result_multiclass.csv
```

### 6.3 单张图片预测

```powershell
python src/predict_one_multiclass.py `
  --image_path data/model_dataset/roi_images/your_image.jpg `
  --checkpoint outputs/multiclass_resnet18_model_dataset/best_resnet18_multiclass.pth
```

输出：

```text
预测类别：无 / 少 / 中 / 多
prob_none
prob_small
prob_medium
prob_large
```

## 7. 瓶体检测组件

该组件用于先训练一个单类别 `bottle` 检测器，再用检测框裁剪统一 ROI。它是独立模块，不会替代或修改现有 ResNet 二分类、四分类训练流程；分类脚本仍然通过 `--image_dir`、`--label_csv` 和 split txt 读取数据。

### 7.1 用 newtrashy 生成单类瓶体检测数据集

`../newtrashy/` 是 Roboflow YOLO 检测数据集，原始类别包含多种瓶型。为了训练统一的瓶体位置检测器，脚本会把所有原始类别统一重映射为：

```text
0 = bottle
```

生成单类别 YOLO 数据集：

```powershell
python src/detection/prepare_newtrashy_bottle_dataset.py `
  --source_dir ../newtrashy `
  --output_dir data/bottle_detection
```

输出结构为：

```text
data/bottle_detection/
├── images/train, val, test
├── labels/train, val, test
└── data.yaml
```

其中 `newtrashy/valid` 会映射为 YOLO 常用的 `val`。

### 7.2 训练瓶体检测器

默认使用 `yolov8n.pt` 作为预训练权重：

```powershell
python src/detection/train_bottle_detector.py `
  --data_yaml data/bottle_detection/data.yaml `
  --model yolov8n.pt `
  --epochs 50 `
  --imgsz 640 `
  --batch 8 `
  --project outputs/bottle_detector `
  --name yolov8n_bottle
```

训练完成后，最佳权重通常位于：

```text
outputs/bottle_detector/yolov8n_bottle/weights/best.pt
```

CPU 可以运行，但 YOLO 训练会比较慢；如果有 NVIDIA GPU，可以通过 `--device 0` 指定 GPU。

### 7.3 使用检测器裁剪统一 ROI

```powershell
python src/detection/crop_roi_with_detector.py \
  --image_dir data/model_dataset/roi_images \
  --label_csv data/model_dataset/annotations/labels.csv \
  --checkpoint outputs/bottle_detector/yolov8n_bottle/weights/best.pt \
  --output_roi_dir data/detector_roi_dataset/roi_images \
  --output_label_csv data/detector_roi_dataset/annotations/labels.csv \
  --split_dir data/model_dataset/splits \
  --output_split_dir data/detector_roi_dataset/splits \
  --conf 0.25 \
  --imgsz 640 \
  --expand_ratio 0.05
```

默认情况下，如果某张图片没有检测到 bottle，会跳过该图片，并同步生成过滤后的 split 文件。若希望无检测结果时使用整图作为 ROI，可以加上：

```powershell
--fallback_full_image
```

生成的 `labels.csv` 字段保持为：

```text
filename,xmin,ymin,xmax,ymax,has_liquid,liquid_level,liquid_class,pose,liquid_type,source
```

### 7.4 使用统一 ROI 继续训练分类模型

二分类：

```powershell
python src/train_binary.py `
  --image_dir data/detector_roi_dataset/roi_images `
  --label_csv data/detector_roi_dataset/annotations/labels.csv `
  --train_txt data/detector_roi_dataset/splits/train.txt `
  --val_txt data/detector_roi_dataset/splits/val.txt `
  --output_dir outputs/binary_resnet18_detector_roi
```

四分类：

```powershell
python src/train_multiclass.py `
  --image_dir data/detector_roi_dataset/roi_images `
  --label_csv data/detector_roi_dataset/annotations/labels.csv `
  --train_txt data/detector_roi_dataset/splits/train.txt `
  --val_txt data/detector_roi_dataset/splits/val.txt `
  --output_dir outputs/multiclass_resnet18_detector_roi
```

如果需要控制输入样本量，可以在二分类或四分类训练时加入：

```powershell
--max_train_samples_per_class 300 `
--max_val_samples_per_class 100
```

也可以控制总量：

```powershell
--max_train_samples 800 `
--max_val_samples 200
```

训练开始时会打印实际使用的样本数和各类别数量。对于四分类增强数据，建议优先使用 `--max_train_samples_per_class`，避免新增 `none` 样本过多导致类别再次失衡。

## 8. 常见问题

### 8.1 找不到数据

确认以下路径存在：

```text
data/model_dataset/annotations/labels.csv
data/model_dataset/roi_images/
data/model_dataset/splits/train.txt
data/model_dataset/splits/val.txt
data/model_dataset/splits/test.txt
```

### 8.2 找不到模型权重

如果运行测试或单图预测，需要先准备 checkpoint：

```text
outputs/binary_resnet18_model_dataset/best_resnet18_binary.pth
outputs/multiclass_resnet18_model_dataset/best_resnet18_multiclass.pth
```

如果没有权重文件，请先运行训练命令。

### 8.3 当前目录错误

所有命令应在 `BottleLiquid` 目录下执行。可以用下面命令检查：

```bash
pwd
```

当前目录中应该能看到：

```text
src/
data/
requirements.txt
README.md
```

### 8.4 CPU 运行较慢

项目可以在 CPU 上运行，只是训练会较慢。若有 GPU，请安装对应 CUDA 版 PyTorch。

## 9. GitHub 上传建议

建议上传：

```text
src/
README.md
requirements.txt
environment.yml
.gitignore
```

不建议上传：

```text
data/model_dataset/roi_images/
outputs/**/*.pth
outputs/**/*.csv
```

数据和模型权重建议通过网盘共享，并在 README 或团队文档中说明下载链接和放置路径。
