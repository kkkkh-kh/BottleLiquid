## 基于人工特征的传统机器学习建模说明

本目录提供不使用 CNN 的瓶内残留液体识别实验代码，覆盖二分类和四分类。



### 一键实验

使用现有 ROI，并按四分类标签分层抽样划分训练集/测试集：

```bash
python3 src/traditional_ml_liquid.py
```

使用数据集中已有 train/val/test 划分：

```bash
python3 src/traditional_ml_liquid.py --use-existing-splits
```

输出目录：

```text
outputs/traditional_ml/
```

主要结果文件：

- `summary.json`：标签校验、样本数、特征维度、最优模型摘要
- `labels_validated.csv`：校验后的标签文件，确保 `has_liquid` 与 `liquid_level/liquid_class` 一致
- `binary_model_comparison.csv`：Logistic、SVM、随机森林、GBDT/XGBoost 的二分类指标
- `level_model_comparison.csv`：四分类模型和两阶段模型指标
- `feature_ablation.csv`：单特征与融合特征的消融实验
- `binary_svm_xgboost_ensemble.csv`：二分类 SVM + XGBoost 概率融合实验结果
- `level_svm_xgboost_ensemble.csv`：四分类 SVM + XGBoost 概率融合实验结果
- `binary_cm_*.csv`：二分类混淆矩阵
- `level_cm_*.csv`：四分类混淆矩阵
- `error_samples.csv`：错误样本清单，并提示从反光、标签遮挡、透明液体、横放、复杂背景等角度复核

### 特征设计

程序提取七类人工特征并拼接为融合特征向量：

1. 灰度统计特征：全局、上下半区、左右半区、下半中心区域的均值、标准差、分位数和上下差异。
2. HSV 颜色直方图特征：H/S/V 通道直方图，以及上下半区饱和度、亮度差异。
3. 边缘密度特征：Sobel 梯度在不同阈值和不同区域中的密度，反映瓶壁、水体边界和纹理强度。
4. 水平液面线特征：在瓶体中心区域搜索水平梯度峰值，记录候选液面位置、强度、上下灰度差。
5. HOG 特征：统计局部梯度方向分布，描述瓶体内部纹理和边界形态。

6. 3x3 区域统计特征：将瓶体 ROI 划分为九宫格，对每个区域分别计算灰度统计、HSV 均值/标准差、边缘密度和梯度统计，用于描述液体在局部空间中的分布差异。
7. 姿态自适应液面线特征：将 ROI 按多个角度旋转，分别搜索候选液面线，并记录最强响应角度、位置、强度和上下灰度差，使液面线特征对倾斜、横放等姿态更鲁棒。

当前完整特征维度为：

```text
gray: 52
hsv: 40
edge: 39
line: 9
adaptive_line: 36
grid: 162
hog: 1980
total: 2318
```

### 标签

二分类：

```text
0 = 无液体
1 = 有液体
```

四分类：

```text
0 = none
1 = small
2 = medium
3 = large
```

### 论文建模表述建议

可将算法流程写为：

ROI 裁剪 -> 尺寸归一化与中值滤波 -> RGB/灰度/HSV 颜色空间转换 -> 人工特征提取 -> 特征向量标准化 -> 分类器训练与模型选择 -> SVM/XGBoost 概率融合。

四分类推广建议采用两阶段模型：

第一阶段判断是否有液体；第二阶段仅对有液体样本判断 `small/medium/large`。这种设计能降低空瓶与液体等级之间的耦合错误，更适合小样本和类别不均衡场景。

### 概率融合实验

程序还实现了 SVM + XGBoost 概率融合：

```text
P = w * P_svm + (1 - w) * P_xgboost
```

其中 `w` 在 0.25、0.40、0.50、0.60、0.75 中搜索。二分类按 F1 选择最优融合方案，四分类按 Macro F1 选择最优融合方案。

融合实验会比较多个有效特征组合：

```text
gray + hsv + edge
gray + hsv + edge + line
gray + hsv + edge + adaptive_line
gray + hsv + edge + grid
gray + hsv + edge + line + adaptive_line + grid
```

对维度较高的组合，程序会用 `SelectKBest(f_classif)` 筛选 96、128、192、256 维候选特征。

对应的最优模型会保存为：

```text
best_binary_ensemble_*.joblib
best_level_ensemble_*.joblib
best_binary_ensemble_*_cols.npy
best_level_ensemble_*_cols.npy
```

本轮新增特征和融合后的代表结果：

```text
二分类最优融合:
feature = gray + hsv + edge + adaptive_line
model = 0.60 * SVM + 0.40 * XGBoost
Accuracy = 0.9447
F1 = 0.9651

四分类最优融合:
feature = gray + hsv + edge + line, SelectKBest(f_classif, k=128)
model = 0.60 * SVM + 0.40 * XGBoost
Accuracy = 0.7972
Macro F1 = 0.7822
```
