import json
import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment
from rapidfuzz import fuzz
import time 
import pandas as pd 
import os
from datetime import datetime
import traceback
import logging 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def f1_score(p, r):
    """Calculates F1 score, handles zero division."""
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)

def calculate_iou(boxA, boxB):
    """Calculates Intersection over Union for two bounding boxes."""
    try:
        boxA = [float(c) for c in boxA]
        boxB = [float(c) for c in boxB]
    except (ValueError, TypeError):
        return 0.0

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0

class ScoreCalculator:
    """
    Calculates scores for the circuit diagram recognition task.
    """
    def __init__(self, iou_threshold=0.5, name_similarity_threshold=80, save_path=None, submit_id=None):
        """
        Initializes the calculator with scoring parameters.
        :param iou_threshold: Minimum IoU for a position match.
        :param name_similarity_threshold: Minimum similarity ratio (0-100) for a name match.
        """
        self.iou_th = iou_threshold
        self.name_th = name_similarity_threshold
        self.submit_id = submit_id
        self.task1_weight = [0.4,0.2,0.4]
        self.task1_task2_weight = [0.6,0.4]

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = save_path

        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.submit_id_save_path = os.path.join(self.save_path, self.submit_id, datetime_str)
        if not os.path.exists(self.submit_id_save_path):
            os.makedirs(self.submit_id_save_path)

        # 初始化日志文件到提交路径
        try:
            log_file_path = os.path.join(self.submit_id_save_path, "run.log")
            file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(log_file_path) for h in logger.handlers):
                logger.addHandler(file_handler)
            logger.info("日志文件已初始化: %s", log_file_path)
        except Exception:
            print(traceback.format_exc())

    def _compute_f1_from_metrics(self, tp, total_pred, total_label):
        """Computes Precision, Recall, and F1 Score from TP, FP, FN counts."""
        p = tp / total_pred if total_pred > 0 else 0
        r = tp / total_label if total_label > 0 else 0
        f1 = f1_score(p, r)
        return p, r, f1

    def _find_component_matches(self, label_components, pred_components):
        """Finds the best matches between label and prediction components based on IoU."""
        valid_matches = []
        pred_idx_to_label_name = {}

        if len(label_components) > 0 and len(pred_components) > 0:
            iou_matrix = np.zeros((len(label_components), len(pred_components)))
            for i, lbl_comp in enumerate(label_components):
                for j, pred_comp in enumerate(pred_components):
                    iou_matrix[i, j] = calculate_iou(lbl_comp.get('Pos'), pred_comp.get('Pos'))

            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            
            for i, j in zip(row_ind, col_ind):
                if iou_matrix[i, j] >= self.iou_th:
                    valid_matches.append((i, j))
                    pred_idx_to_label_name[j] = label_components[i].get('Component')
        
        return {
            "valid_matches": valid_matches,
            "pred_idx_to_label_name": pred_idx_to_label_name
        }

    def _calculate_s1_tp(self, label_components, pred_components, valid_matches):
        """Calculates TP for position and name based on the found matches."""
        tp_pos = len(valid_matches)
        return tp_pos

    def _calculate_s2_metrics(self, label_components, pred_components, valid_matches):
        """Calculates I/O (S2) metrics for a single image based on established matches."""
        tp_io = 0
        for label_idx, pred_idx in valid_matches:
            lbl_io = label_components[label_idx].get('I_O')
            pred_io = pred_components[pred_idx].get('I_O')
            if lbl_io and pred_io and lbl_io.get('input') == pred_io.get('input') and lbl_io.get('output') == pred_io.get('output'):
                tp_io += 1
        return tp_io

    def _calculate_s3_metrics(self, label_components, pred_components, pred_idx_to_label_name, image_key):
        """Calculates connection (S3) metrics for a single image based on component-level scoring."""
        # 构建标签组件的连接字典 - 每个组件名映射到其连接信息
        label_component_connections = {}
        for comp in label_components:
            name = comp.get('Component')
            if name and comp.get('Connection'):
                connections = comp['Connection']
                # 标准化连接信息：排序输入和输出列表以便比较
                input_list = sorted(connections.get('input', []))
                output_list = sorted(connections.get('output', []))
                label_component_connections[name] = {
                    'input': input_list,
                    'output': output_list
                }

        # 构建预测组件的连接字典
        pred_component_connections = {}
        for pred_comp_idx, pred_comp in enumerate(pred_components):
            true_source_name = pred_idx_to_label_name.get(pred_comp_idx)
            if true_source_name and pred_comp.get('Connection'):
                connections = pred_comp['Connection']
                
                # 将预测的组件名转换为真实的标签名
                mapped_input_list = []
                mapped_output_list = []
                
                pred_name_to_idx = {comp.get('Component'): i for i, comp in enumerate(pred_components)}
                
                for pred_target_name in connections.get('input', []):
                    pred_target_idx = pred_name_to_idx.get(pred_target_name)
                    true_target_name = pred_idx_to_label_name.get(pred_target_idx)
                    if true_target_name:
                        mapped_input_list.append(true_target_name)
                
                for pred_target_name in connections.get('output', []):
                    pred_target_idx = pred_name_to_idx.get(pred_target_name)
                    true_target_name = pred_idx_to_label_name.get(pred_target_idx)
                    if true_target_name:
                        mapped_output_list.append(true_target_name)
                
                # 标准化连接信息：排序输入和输出列表以便比较
                pred_component_connections[true_source_name] = {
                    'input': sorted(mapped_input_list),
                    'output': sorted(mapped_output_list)
                }

        # 按组件维度计算分数
        tp_components = 0  # 连接完全正确的组件数量
        total_pred_components_with_connections = len(pred_component_connections)
        total_label_components_with_connections = len(label_component_connections)
        
        # 检查每个有连接信息的组件
        for component_name in label_component_connections:
            if component_name in pred_component_connections:
                label_conn = label_component_connections[component_name]
                pred_conn = pred_component_connections[component_name]
                
                # 检查该组件的输入和输出连接是否完全匹配
                if (label_conn['input'] == pred_conn['input'] and 
                    label_conn['output'] == pred_conn['output']):
                    tp_components += 1
        
        return tp_components, total_pred_components_with_connections, total_label_components_with_connections


    def calculate_task1_scores(self,label_data,pred_data,result):
        """
        Calculates scores for all images and aggregates them for a final score.
        """
        per_image_scores = {}
        all_image_keys = set(label_data.keys()) & set(pred_data.keys())

        for image_key in all_image_keys:
            try:
                label_components = label_data.get(image_key, {}).get('task1', [])
                pred_components = pred_data.get(image_key, {}).get('task1', [])

                # Find component matches first, this is the basis for all scores
                matches = self._find_component_matches(label_components, pred_components)
                valid_matches = matches['valid_matches']
                pred_idx_to_label_name = matches['pred_idx_to_label_name']

                # S1
                tp_pos = self._calculate_s1_tp(label_components, pred_components, valid_matches)
                _, _, sp = self._compute_f1_from_metrics(tp_pos, len(pred_components), len(label_components))
                s1 = sp

                # S2
                tp_io = self._calculate_s2_metrics(label_components, pred_components, valid_matches)
                total_pred_io = sum(1 for c in pred_components if c.get('I_O'))
                total_label_io = sum(1 for c in label_components if c.get('I_O'))
                _, _, s2 = self._compute_f1_from_metrics(tp_io, total_pred_io, total_label_io)

                # S3
                tp_conn, total_pred_conn, total_label_conn = self._calculate_s3_metrics(label_components, pred_components, pred_idx_to_label_name, image_key)
                _, _, s3 = self._compute_f1_from_metrics(tp_conn, total_pred_conn, total_label_conn)

                # Store per image
                per_image_scores[image_key] = {
                    "S1": s1, "S2": s2, "S3": s3,
                    "task1_score": self.task1_weight[0]* s1 + self.task1_weight[1] * s2 + self.task1_weight[2] * s3
                }
            except:
                logger.error("计算任务1分数时发生异常: label_data=%s, pred_data=%s,异常信息=%s", label_data, pred_data,str(traceback.format_exc()))
                continue
        result["per_image"].update(per_image_scores)
        return result
    
    def calculate_task2_scores(self, label_data, pred_data, result):
        """
        Calculates and adds Task 2 scores to the results dictionary.
        """
        all_image_keys = set(label_data.keys()) | set(pred_data.keys())

        for image_key in all_image_keys:
            try:
                label_qa_list = label_data.get(image_key, {}).get('task2', [])
                pred_qa_list = pred_data.get(image_key, {}).get('task2', [])

                # Use the question as a key for matching (case-insensitive, stripped)
                label_map = {
                    item['question'].strip().lower(): str(item.get('answer', '')).strip().lower()
                    for item in label_qa_list if 'question' in item
                }
                
                total_questions = len(label_map)
                if total_questions == 0:
                    if image_key in result.get("per_image", {}):
                        result["per_image"][image_key]["task2_score"] = 0.0
                    continue

                correct_count = 0
                # Iterate through predictions and score them against the label map
                for pred_item in pred_qa_list:
                    pred_question = pred_item.get('question', '').strip().lower()
                    pred_answer = str(pred_item.get('answer', '')).strip().lower()
                    
                    # Check if this question exists in the label and if the answer matches
                    if pred_question in label_map and label_map[pred_question] == pred_answer:
                        correct_count += 1
                
                
                # Update the per-image results dictionary
                if image_key not in result.get("per_image", {}):
                    result["per_image"][image_key] = {}
                result["per_image"][image_key].update({
                    "task2_correct": correct_count,
                    "task2_total": total_questions})
            except:
                logger.error("计算任务2分数时发生异常: label_data=%s, pred_data=%s,异常信息=%s", label_data, pred_data,str(traceback.format_exc()))
                continue
        return result
    
    def calculate_final_score(self,result):
        task1_score = 0
        task2_correct = 0
        logger.info("计算最终分数: result=%s", result)
        for image_key, image_value in result["per_image"].items():
            task1_score += image_value.get("task1_score", 0)
            task2_correct += image_value.get("task2_correct", 0)
        task1_score = task1_score/len(result["per_image"])
        task2_score = task2_correct/(2*len(result["per_image"]))
        final_score = task1_score * self.task1_task2_weight[0] + task2_score * self.task1_task2_weight[1]
        return {"task1_score": task1_score, "task2_score": task2_score, "final_score": final_score}
    
    def init_result(self,label_data):  
        result = {"per_image": {}, "final_score": {}}
        for image_key in label_data.keys():
            result["per_image"][image_key] = {"S1": 0, "S2": 0, "S3": 0, "task1_score": 0, "task2_correct": 0, "task2_total": 0}

        print(result)
        return result

    def calculate_scores(self, label_data, pred_data):
        result=self.init_result(label_data)
        result= self.calculate_task1_scores(label_data,pred_data,result)
        result.update(self.calculate_task2_scores(label_data,pred_data,result))
       
        ##返回分数
        final_score = self.calculate_final_score(result)
        result["final_score"] = final_score

        return result

    def get_data_from_path(self,path):
        json_list = {}
        for file in os.listdir(path):
            if file.endswith('.json'):  
                with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                json_list[file] = data
        return json_list
    
    def post_process(self,results):
        # save_path = os.path.join(self.submit_id_save_path, "results.csv")
        # pd_results = pd.DataFrame.from_dict(results['per_image'],orient='index').sort_index().reset_index().rename(columns={'index':'image_name'})
        # pd_results.to_csv(save_path, index=False)

        ##json
        save_path = os.path.join(self.submit_id_save_path, "results.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logger.info("结果已保存: %s", save_path)
        
        logger.info("结果已保存: %s", results)
        
        # 生成Markdown格式的测试报告
        self.generate_report(results)

    def generate_report(self, results):
        """
        生成Markdown格式的测试报告
        """
        import math
        
        # 整理每张图片的数据格式
        reports = []
        for image_name, image_result in results['per_image'].items():
            # 提取图片编号（假设文件名格式为 xxx.json）
            case_id = image_name.replace('.json', '') if image_name.endswith('.json') else image_name
            
            report = {
                '测例编号': case_id,
               
                'S1位置识别': round(image_result.get('S1'), 3),
                'S2输入输出': round(image_result.get('S2'), 3),
                'S3连接关系': round(image_result.get('S3'), ),
                '任务1得分': round(image_result.get('task1_score'), 3),
                '任务2正确数': f"{image_result.get('task2_correct')}"
            }
            reports.append(report)
        
        # 按测例编号排序
        def extract_case_number(report):
            case_id = report['测例编号']
            # 尝试提取数字部分进行排序
            import re
            numbers = re.findall(r'\d+', str(case_id))
            return int(numbers[0]) if numbers else 0
        
        reports.sort(key=extract_case_number)
        
        # 设置表头
        headers = ["测例编号", "S1位置识别", "S2输入输出", "S3连接关系","任务1得分",  "任务2正确数"]
        
        # 创建表头行和分隔行
        header_line = " | ".join(headers)
        separator_line = " | ".join(["---"] * len(headers))
        
        # 创建数据行
        rows = []
        for report in reports:
            row = []
            for header in headers:
                row.append(str(report[header]))
            rows.append(" | ".join(row))
        
        # 计算总体统计
        final_scores = results.get('final_score', {})
        task1_avg = final_scores.get('task1_score')
        task2_avg = final_scores.get('task2_score')
        final_score = final_scores.get('final_score')
        
        # 生成Markdown报告内容
        markdown_table = f"""# 电路图识别测试报告

## 详细测试结果

{header_line}
{separator_line}
""" + "\n".join(rows) + f"""

**说明:** 
- S1: 位置识别准确度 (权重: {self.task1_weight[0]})
- S2: 输入输出识别准确度 (权重: {self.task1_weight[1]}) 
- S3: 连接关系识别准确度 (权重: {self.task1_weight[2]})
- 任务2: 问答正确数/总问题数

## 综合测试结果

| 任务1平均得分 | 任务2平均得分 | 最终加权得分 |
| --- | --- | --- |
| {task1_avg:.3f} | {task2_avg:.3f} | {final_score:.3f} |

**说明:**
- 总测试图片数: {len(results['per_image'])}
- 任务1加权公式: S1×{self.task1_weight[0]} + S2×{self.task1_weight[1]} + S3×{self.task1_weight[2]}
- 最终得分公式: 任务1得分×{self.task1_task2_weight[0]} + 任务2得分×{self.task1_task2_weight[1]}
"""
        
        # 保存报告
        report_path = os.path.join(self.submit_id_save_path, 'report.md')
        with open(report_path, "w", encoding="utf-8") as file:
            file.write(markdown_table)
        
        # print(f"测试报告已保存在 {report_path}")
        logger.info("测试报告已保存在 %s", report_path)
        return report_path

    def run(self,label_path,pred_path):
        ##获取label 和 pred_data
        logger.info("开始评分: label_path=%s, pred_path=%s, submit_id_save_path=%s", label_path, pred_path, self.submit_id_save_path)
        st = time.time()
        try:
            label_data = self.get_data_from_path(label_path)
            pred_data  = self.get_data_from_path(pred_path)
            ##计算分数
            results = self.calculate_scores(label_data,pred_data)
            self.post_process(results)
        except:
            logger.error("评分时发生异常: label_path=%s, pred_path=%s,异常信息=%s", label_path, pred_path,str(traceback.format_exc()))
            return None
        cost_time = time.time() - st
        logger.info("评分完成,用时%s秒。", cost_time)
        return results

def main():
    parser = argparse.ArgumentParser(description="计算电路图识别任务的分数")
    parser.add_argument("label_file", help="标签文件路径")
    parser.add_argument("predict_file", help="结果文件路径")
    parser.add_argument("--save_path", default="results", help="保存结果的文件路径")
    parser.add_argument("--submit_id", default="testid", help="提交id")
    args = parser.parse_args()
    calculator = ScoreCalculator(iou_threshold=0.5, name_similarity_threshold=80, save_path=args.save_path, submit_id=args.submit_id)
    calculator.run(args.label_file, args.predict_file)

if __name__ == "__main__":
    main() 