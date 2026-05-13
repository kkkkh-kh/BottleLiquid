下面是当前 `BottleLiquid/data/model_dataset` 的数据集说明，可以直接写进报告或 README。

**一、数据集总体说明**

当前最终整理后的模型数据集位于：

```text
BottleLiquid/data/model_dataset/
```

这是目前推荐用于训练与测试的统一数据集。它已经把前期不同来源、不同格式的数据统一整理为同一种标注格式，并统一裁剪为瓶体 ROI 图片。

总样本数：

```text
1442 张 ROI 图片
```

任务标签包括两种层次：

```text
二分类：has_liquid
0 = 无液体
1 = 有液体

四分类：liquid_class / liquid_level
0 = none   无
1 = small  少
2 = medium 中
3 = large  多
```

**二、数据来源说明**

当前数据集由 5 类来源整合而成：

| 来源标识 | 数量 | 原始来源 | 说明 |
|---|---:|---|---|
| `folder_yolo` | 486 | `Full Water level`、`Half water level`、`Overflowing` | 根据文件夹名生成弱标签，并使用 YOLO 自动检测瓶体框 |
| `coco` | 290 | `data/train/_annotations.coco.json` | Roboflow COCO 标注，包含瓶体和液体区域，通过液体面积比例推导水量等级 |
| `newtrashy_none` | 150 | `newtrashy` | 人工审查认为可作为空瓶样本，从 YOLO segmentation 中随机抽取瓶体 ROI，标为 `none` |
| `data_data` | 110 | `data/data` | YOLO 格式人工标注，包含瓶体框和 `small/none/large/medium` 类别框 |
| `database` | 406 | `database/label.csv` | 标准 CSV 标注，已有瓶体框和液体量等级 |

合计：

```text
486 + 290 + 150 + 110 + 406 = 1442
```

**三、类别分布**

四分类分布：

```text
large:   540
medium:  379
none:    324
small:   199
```

二分类分布：

```text
有液体: 1118
无液体:  324
```

其中：

```text
有液体 = small + medium + large
无液体 = none
```

**四、数据集内部结构**

当前统一数据集目录结构：

```text
BottleLiquid/data/model_dataset/
├── annotations/
│   └── labels.csv
│
├── roi_images/
│   ├── coco_xxx.jpg
│   ├── database_xxx.jpg
│   ├── data_data_xxx.jpg
│   ├── folder_xxx.jpeg
│   ├── newtrashy_none_xxx.jpg
│   └── ...
│
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

各部分含义：

```text
annotations/labels.csv
```

统一标注文件。模型训练读取该文件中的 `filename` 和标签列。

```text
roi_images/
```

裁剪后的瓶体 ROI 图片。ResNet18 分类模型实际输入的是这里的图片。

```text
splits/
```

训练集、验证集、测试集划分文件。每个 txt 文件每行是一个 ROI 图片文件名。

**五、划分情况**

```text
train: 1009
val:    216
test:   217
```

各 split 的四分类分布：

```text
train:
large     378
medium    265
none      227
small     139

val:
large      81
medium     57
none       48
small      30

test:
large      81
medium     57
none       49
small      30
```

这个划分是按 `liquid_class` 分层抽样得到的，保证训练、验证、测试集中各类别比例相对一致。

**六、labels.csv 字段说明**

当前字段为：

```text
filename,xmin,ymin,xmax,ymax,has_liquid,liquid_level,liquid_class,pose,liquid_type,source
```

字段含义：

| 字段 | 含义 |
|---|---|
| `filename` | ROI 图片文件名，对应 `roi_images/` 下的图片 |
| `xmin,ymin,xmax,ymax` | 保留的瓶体框字段，用于兼容和溯源 |
| `has_liquid` | 二分类标签，`0` 无液体，`1` 有液体 |
| `liquid_level` | 四分类文本标签：`none/small/medium/large` |
| `liquid_class` | 四分类数字标签：`0/1/2/3` |
| `pose` | 姿态字段，目前统一为 `unknown`，不参与训练 |
| `liquid_type` | 液体类型，主要为 `water/none` |
| `source` | 样本来源，用于追溯数据来自哪个原始数据集 |

需要注意：当前 `model_dataset` 已经是 ROI 数据集，训练时不会再根据 `xmin/ymin/xmax/ymax` 裁剪图片；模型直接读取 `roi_images/filename`。

**七、推荐训练路径**

二分类有无液体：

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

四分类液体量：

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

**八、简短总结**

当前 `model_dataset` 是统一后的最终训练数据集。它整合了弱标签、COCO 标注、人工标注、空瓶增强和标准 CSV 数据，统一成 ROI 图片 + CSV 标签 + split 文件的结构。模型训练和测试只需要使用：

```text
data/model_dataset/roi_images/
data/model_dataset/annotations/labels.csv
data/model_dataset/splits/
```