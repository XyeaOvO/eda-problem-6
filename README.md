# EDA Problem 6 — 电路系统框图识别与解析

## 项目简介

本项目是一个针对**电路系统框图（Block Diagram）**的自动识别与解析竞赛题目。参赛者需要开发算法，对电路系统框图图片进行以下两项任务的处理：

- **任务1（Task 1）**：识别框图中的各个组件，并解析其位置、输入输出属性及组件间的连接关系。
- **任务2（Task 2）**：根据框图内容，回答与电路相关的单选题或填空题。

> 详细的赛题文档见 [`docs/电路系统框图识别与解析.pdf`](docs/电路系统框图识别与解析.pdf)

---

## 目录结构

```
eda-problem-6/
├── datasets/
│   ├── Public/                     # 公开数据集
│   │   ├── 1000_images/            # ~1000 张训练用电路框图图片
│   │   └── benchmark/
│   │       ├── images/             # 20 张评测图片
│   │       └── jsons/              # 20 张图片对应的标注 JSON（含 task1 & task2 答案）
│   └── Hidden/                     # 隐藏测试集（评测时使用）
│       ├── images/                 # 80 张评测图片
│       ├── jsons/                  # 80 张图片对应的完整标注 JSON
│       └── task2_question/         # 80 张图片对应的 task2 问题文件（不含答案）
├── docs/
│   └── 电路系统框图识别与解析.pdf   # 赛题说明文档
├── scores.py                       # 评分脚本
└── readme_submission_and_score.md  # 提交与评分说明
```

---

## 数据格式

### 标注 JSON 结构（Task 1 + Task 2）

每张图片对应一个 JSON 文件，结构如下：

```json
{
  "task1": [
    {
      "Component": "组件名称",
      "Pos": [x1, y1, x2, y2],
      "I_O": { "input": 0, "output": 1 },
      "Connection": {
        "input": ["上游组件名"],
        "output": ["下游组件名"]
      }
    }
  ],
  "task2": [
    {
      "type": "multiple_choice",
      "question": "题目文本",
      "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "answer": "A"
    },
    {
      "type": "fill_in_the_blank",
      "question": "题目文本",
      "answer": "答案文本"
    }
  ]
}
```

- `Pos`：边界框坐标，格式为 `[x_min, y_min, x_max, y_max]`
- `I_O.input` / `I_O.output`：整数，表示该组件的输入/输出端数量
- `Connection.input` / `Connection.output`：列表，存放与该组件相连的其他组件名称

---

## 评分规则

### 任务1 评分（三个子指标）

| 指标 | 说明 | 权重 |
|------|------|------|
| **S1** 位置识别 | 以 IoU ≥ 0.5 判断组件框匹配，用 F1 评分 | 0.4 |
| **S2** 输入输出识别 | 在位置匹配的基础上，判断 `I_O` 是否正确，用 F1 评分 | 0.2 |
| **S3** 连接关系识别 | 在位置匹配的基础上，判断组件间连接（输入/输出列表）是否完全正确，用 F1 评分 | 0.4 |

**任务1得分** = S1 × 0.4 + S2 × 0.2 + S3 × 0.4

### 任务2 评分

- 每张图片包含 2 道题（1 道单选 + 1 道填空），答案完全匹配得分。
- **任务2得分** = 全部图片答对总数 / (2 × 图片总数)

### 最终得分

**最终得分 = 任务1得分 × 0.6 + 任务2得分 × 0.4**

### 耗时加分

- 总耗时排名前 10%：+5 分
- 总耗时排名前 30%：+3 分
- 总耗时排名前 60%：+1 分

---

## 提交要求

参赛者需提供一个**统一入口脚本 `entry.py`**，仅接受两个参数：

```bash
python entry.py --image_path /path/to/images --output_path /path/to/output
```

- `image_path`：评测图片（或图片目录）的路径
- `output_path`：预测结果输出目录，**每张图片对应一个同名 JSON 文件**

输出 JSON 的结构需与公开数据集的标注 JSON 完全一致（含 `task1` 与 `task2` 两个字段），文件编码为 **`utf-8-sig`**。

---

## 评分脚本使用

使用 `scores.py` 在本地计算分数：

```bash
python scores.py \
  --label_file /path/to/labels/ \
  --predict_file /path/to/predictions/ \
  --save_path /path/to/eval_outputs \
  --submit_id 123456
```

脚本输出目录结构：

```
{save_path}/{submit_id}/{时间戳}/
  ├── results.json   # 评分明细与汇总
  ├── report.md      # 可视化 Markdown 报告
  └── run.log        # 运行日志
```

---

## 快速开始

1. 查阅赛题文档：[`docs/电路系统框图识别与解析.pdf`](docs/电路系统框图识别与解析.pdf)
2. 阅读提交规范：[`readme_submission_and_score.md`](readme_submission_and_score.md)
3. 使用公开数据集进行算法开发：`datasets/Public/`
4. 在本地使用 `scores.py` 对预测结果进行评分验证
5. 通过平台提交最终 `entry.py` 及模型
