## 提交与评分说明

### 提交要求
- **通过平台提交**：请在平台完成所有算法与模型的提交。
- **统一入口脚本**：需提供单一入口脚本，且仅接受两个入参：`image_path` 与 `output_path`。
  - `image_path`：评测时输入图片（或图片目录）的路径
  - `output_path`：算法需要将预测结果写入到该路径，`每一个图片对应一个结果json文件，且与图片名字同名`。

#### 入口脚本规范
- 入口脚本示例调用：
```bash
python entry.py --image_path /path/to/images --output_path /path/to/output
```
- 脚本需在 `output_path` 目录下生成评测所需的预测结果 JSON（`每一个图片对应一个结果json文件，且与图片名字同名`）。
- 运行期间不得对 `label_file` 进行写入修改。

##### 预测结果 JSON 要求
- 编码：`utf-8-sig`
- 结构：与公开 20 条数据集的 JSON 完全一致。
- 完整性：
  - 覆盖全部待评测样本
  - 不缺少必须字段，不更改字段含义

##### 常见问题与注意事项
- 请确保文件编码为 `utf-8-sig`，否则可能导致解析异常或评分失败。
- `submit_id` 用于区分不同提交与结果版本，建议与平台保持一致，便于追溯。
- 若预测过程会生成中间文件，请不要写入到 `save_file` 指定目录，以避免与评分输出混淆。

### 评分与排名计算
#### 算法分数
- **评分脚本**：使用 `scores.py` 进行分数计算与可视化报告生成。
- **输入参数**：
  - `label_file`：数据集标签文件路径
  - `predict_file`：模型预测结果 JSON 文件路径
  - `save_file`：分数与报告保存的根路径
  - `submit_id`：提交 ID（建议与平台提交 ID 保持一致）
- **运行方式（示例）**：
```bash
python scores.py \
  --label_file /path/to/labels.json \
  --predict_file /path/to/predict.json \
  --save_file /path/to/eval_outputs \
  --submit_id 123456
```
- **脚本输出目录结构**：
```
{save_file}/{submit_id}/{时间戳}/
  ├─ result.json   # 评分明细与汇总
  └─ report.md     # 可视化报告（Markdown）
```
  - 其中 `{时间戳}` 由脚本按运行时间自动生成。
#### 模型耗时分数
- 根据系统记录的总耗时进行排序，然后根据耗时占比进行加分。前10%耗时占比加5分，前30%耗时占比加3分，前30%耗时占比加1分。




