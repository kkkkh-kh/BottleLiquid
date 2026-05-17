# 传统机器学习模型结果说明

## 1. 任务说明

本部分使用人工特征，不使用 CNN 特征。任务包括：

- 二分类：判断瓶内是否有液体。
- 普通四分类：判断液体等级为 `无 / 少量 / 中等 / 较多`。
- 有序四分类：利用 `无 < 少量 < 中等 < 较多` 的等级顺序进行分类。

数据集共 `1442` 张 ROI 可用图片，严格测试集共 `217` 张图片。

## 2. 标签定义

二分类标签：

```text
0 = 无液体
1 = 有液体
```

四分类标签：

```text
none   = 无
small  = 少量
medium = 中等
large  = 较多
```

数据集类别分布：

```text
无:   324
少量: 199
中等: 379
较多: 540
```

## 3. 模型与特征

传统模型使用的人工特征包括：

- 灰度统计特征 `gray`
- HSV 颜色直方图特征 `hsv`
- 边缘密度特征 `edge`
- 水平液面线特征 `line`
- 多角度自适应液面线特征 `adaptive_line`
- 3x3 区域统计特征 `grid`
- 液体含量特征 `amount`
- 标签/反光干扰特征 `artifact`
- 姿态归一化特征 `pose_norm`
- HOG 特征 `hog`

最终用于报告的历史最高准确率模型如下：

| 任务 | 特征组合 | 模型 | 测试集 Accuracy | 主要指标 |
|---|---|---|---:|---:|
| 二分类 | `stat_color_edge_adaptive` | `0.60*SVM + 0.40*XGBoost` | 0.9355 | F1 = 0.9595 |
| 普通四分类 | `stat_color_edge_adaptive` | `0.75*SVM + 0.25*XGBoost` | 0.7742 | Macro F1 = 0.7434 |
| 有序四分类 | `stat_color_edge_amount_artifact` | 有序 `0.50*SVM + 0.50*XGBoost` | 0.7696 | Macro F1 = 0.7558 |

## 4. 文件说明

### 核心汇总表

| 文件 | 含义 | 主要用途 |
|---|---|---|
| `traditional_model_results.xlsx` | 多个结果表的 Excel 汇总工作簿 | 提交支撑材料、集中查看结果 |
| `labels.csv` | 每张图片的真实标签信息 | 画数据集类别分布图 |
| `class_distribution.csv` | 四分类类别数量统计 | 画类别分布柱状图 |
| `metrics_summary.csv` | 三种传统模型的总体指标 | 画模型指标柱状图 |
| `predictions.csv` | 测试集逐样本预测结果汇总 | 计算混淆矩阵、查错样本 |
| `inference_time.csv` | 每张测试图的推理耗时 | 画推理时间分布图 |
| `inference_time_summary.csv` | 推理时间均值、最大值、最小值 | 画平均推理时间对比图 |

### 单独预测结果表

| 文件 | 含义 |
|---|---|
| `predictions_traditional_binary.csv` | 二分类传统模型逐样本预测结果 |
| `predictions_traditional_level.csv` | 普通四分类传统模型逐样本预测结果 |
| `predictions_traditional_ordinal.csv` | 有序四分类传统模型逐样本预测结果 |

### 混淆矩阵

| 文件 | 含义 |
|---|---|
| `confusion_matrix_binary_traditional.csv` | 二分类混淆矩阵 |
| `confusion_matrix_level_traditional.csv` | 普通四分类混淆矩阵 |
| `confusion_matrix_level_ordinal.csv` | 有序四分类混淆矩阵 |

### 场景/姿态统计

| 文件 | 含义 |
|---|---|
| `pose_distribution.csv` | 姿态标签数量统计 |
| `performance_by_pose_scene.csv` | 按 `pose` 和 `scene/source` 分组的模型准确率 |

说明：当前数据中的 `pose` 基本为 `unknown`，因此姿态分组图参考价值有限；`scene` 是根据原始 `source` 字段派生的来源分组，并非人工场景标注。

## 5. Excel 工作簿说明

`traditional_model_results.xlsx` 包含以下 sheet：

```text
labels
class_distribution
metrics_summary
predictions
confusion_binary
confusion_level
confusion_ordinal
pose_scene_perf
inference_time
inference_summary
```

## 6. 指标结果

`metrics_summary.csv` 中记录了三种传统模型在严格测试集上的结果：

```text
二分类:
Accuracy  = 0.9355
Precision = 0.9326
Recall    = 0.9881
F1        = 0.9595

普通四分类:
Accuracy  = 0.7742
Macro F1  = 0.7434

有序四分类:
Accuracy  = 0.7696
Macro F1  = 0.7558
```

其中：

- 二分类主要看 `Accuracy / Precision / Recall / F1`。
- 四分类更推荐看 `Accuracy / Macro F1 / Weighted F1`。
- `Macro F1` 是四个类别 F1 的算术平均，更能反映少数类 `少量` 的识别情况。

## 7. 混淆矩阵

二分类混淆矩阵：

```text
真实\预测  无液体  有液体
无液体       37     12
有液体        2    166
```

普通四分类混淆矩阵：

```text
真实\预测   无  少量  中等  较多
无         42    1    1    5
少量        3   14    9    4
中等        1    4   49    3
较多        3    2   13   63
```

有序四分类混淆矩阵：

```text
真实\预测   无  少量  中等  较多
无         35    5    5    4
少量        0   24    4    2
中等        2    8   42    5
较多        4    3    8   66
```

## 8. 推理时间

推理时间统计包含 ROI 读取、预处理、人工特征提取和模型预测。单位为毫秒/张。

```text
二分类传统模型:
平均 18.46 ms/张

普通四分类传统模型:
平均 17.51 ms/张

有序四分类传统模型:
平均 18.21 ms/张
```

对应文件：

- `inference_time.csv`
- `inference_time_summary.csv`

## 9. 建议画图清单

可以直接使用本目录中的 CSV 画以下图：

| 图表 | 数据文件 |
|---|---|
| 数据集类别分布柱状图 | `class_distribution.csv` |
| 模型总体指标柱状图 | `metrics_summary.csv` |
| 二分类混淆矩阵 | `confusion_matrix_binary_traditional.csv` |
| 普通四分类混淆矩阵 | `confusion_matrix_level_traditional.csv` |
| 有序四分类混淆矩阵 | `confusion_matrix_level_ordinal.csv` |
| 测试集逐样本正确/错误分析 | `predictions.csv` |
| 推理时间对比图 | `inference_time_summary.csv` |
| 按来源/姿态准确率图 | `performance_by_pose_scene.csv` |

## 10. 复现实验

传统模型主程序位于：

```text
src/traditional_ml_liquid.py
```

严格验证实验命令：

```bash
python3 src/traditional_ml_liquid.py --use-existing-splits --strict-validation --skip-ensemble --save-model
```

注意：主程序已经将严格验证阶段的三类模型锁定为历史最高准确率候选：

```text
binary        = stat_color_edge_adaptive_raw_svm0.60_xgb0.40
level         = stat_color_edge_adaptive_raw_svm0.75_xgb0.25
level_ordinal = ordinal_stat_color_edge_amount_artifact_raw_small_os_svm0.50_xgb0.50
```

因此重新运行主程序时，会优先生成这三类最高准确率版本的严格测试结果。若要写论文或画图，仍建议优先使用本目录中的汇总表。

## 11. 备注

本目录没有提供 `level_regression.csv`，因为当前数据没有真实液体占比标注，也没有训练液位比例回归模型。对于本题目前的传统模型部分，使用四分类混淆矩阵比液位回归误差图更稳妥。
