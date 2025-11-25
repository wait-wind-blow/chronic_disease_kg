import pandas as pd
import json
import requests
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

# ================= 配置 =================
MODEL_NAME = "qwen3:8b"  # 确保和你 run_extraction_ollama.py 用的一样
OLLAMA_URL = "http://localhost:11434/api/chat"
MAX_WORKERS = 4  # 并发数

# 路径配置
BASE_DIR = Path(__file__).resolve().parent.parent
GOLD_FILE = BASE_DIR / "data" / "gold_standard" / "gold_standard_to_annotate.xlsx"
# 如果你用的是 CSV，请取消下面这行的注释并修改文件名
# GOLD_FILE = BASE_DIR / "data" / "gold_standard" / "gold_standard_to_annotate.csv"

OUTPUT_REPORT = BASE_DIR / "data" / "processed" / "evaluation_report.xlsx"

# ================= Prompt (保持与抽取脚本一致) =================
# 这里复用了你最新的双语思维链 Prompt，确保评估的是同一个逻辑
SYSTEM_PROMPT = """
你是一位精通医学的知识图谱专家。你的任务是处理英文医学文本，并完成“翻译-抽取-对齐”工作。

请严格按照以下三个步骤思考并输出：

### 步骤 1：全文翻译 (Translation)
- 将输入的英文摘要翻译成**专业、流畅的中文**。

### 步骤 2：实体抽取与对齐 (Entity Extraction & Alignment)
- 从**英文原文**中提取关键医学实体。
- **实体类型 (Type) 仅限：** "Disease", "Chemical", "Gene", "Factor"

### 步骤 3：三元组抽取 (Triple Extraction)
- 基于**英文实体**构建关系三元组。
- **关系类型 (Relation) 仅限：** "Treat", "Prevent", "Cause", "Complicate", "Positive_associated", "Negative_associated", "Associated"

### 输出格式要求 (JSON)
必须输出为**唯一的 JSON 对象**，格式如下：
{
  "translation": "中文翻译...",
  "entities": [
    {"name": "Type 2 Diabetes", "zh_name": "2型糖尿病", "type": "Disease"}
  ],
  "triples": [
    {"head": "Metformin", "relation": "Treat", "tail": "Type 2 Diabetes", "confidence": 0.95}
  ]
}
"""

USER_TEMPLATE = """
请处理以下文本：

[Input Text]
"{TEXT}"

[Your Answer]
(Ensure valid JSON only)
"""


# ================= 工具函数 =================

def normalize_str(s):
    """标准化字符串：小写，去首尾空格"""
    if not isinstance(s, str):
        return ""
    return s.strip().lower()


def parse_gold_json(json_str, data_type="entity"):
    """
    解析金标准里的 JSON 字符串。
    兼容你提供的格式：
    Entities: {"Medical_Entities": [...]}
    Triples: [...]
    """
    if not isinstance(json_str, str) or not json_str.strip():
        return []

    try:
        data = json.loads(json_str)

        if data_type == "entity":
            # 你的金标准实体是放在 "Medical_Entities" 里的
            if isinstance(data, dict):
                return data.get("Medical_Entities", [])
            elif isinstance(data, list):
                return data  # 兼容列表格式

        elif data_type == "triple":
            # 你的金标准三元组直接是一个 list
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return data.get("triples", [])  # 兼容 dict 格式

    except Exception as e:
        print(f"⚠️ JSON 解析失败: {e}")
        return []
    return []


def call_ollama(text, row_idx):
    """调用模型"""
    payload = {
        "model": MODEL_NAME,
        "format": "json",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(TEXT=text)},
        ],
        "stream": False,
        "options": {"temperature": 0.0}  # 评估时温度设为 0，保证结果可复现
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if resp.status_code == 200:
            return row_idx, resp.json()["message"]["content"]
    except:
        pass
    return row_idx, None


def calculate_f1(gold_set, pred_set):
    """计算 P, R, F1"""
    tp = len(gold_set.intersection(pred_set))
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1, tp, fp, fn


# ================= 主流程 =================

def main():
    # 1. 读取金标准数据
    print(f"正在读取金标准文件: {GOLD_FILE}")
    try:
        if GOLD_FILE.suffix == '.csv':
            df = pd.read_csv(GOLD_FILE)
        else:
            df = pd.read_excel(GOLD_FILE)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    print(f"共加载 {len(df)} 条测试数据。开始评估...")

    results = []

    # 2. 并发调用模型
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(call_ollama, row["text"], idx): idx
            for idx, row in df.iterrows()
        }

        for future in tqdm(as_completed(future_to_idx), total=len(df)):
            idx, llm_output = future.result()
            row = df.iloc[idx]

            # --- 解析 Gold Standard ---
            gold_ents_raw = parse_gold_json(row.get("human_entities", "[]"), "entity")
            gold_tris_raw = parse_gold_json(row.get("human_triples", "[]"), "triple")

            # 转换为集合用于比较 (标准化为小写)
            # Entity: (name, type)
            gold_ent_set = set()
            for e in gold_ents_raw:
                gold_ent_set.add((normalize_str(e.get("name")), normalize_str(e.get("type"))))

            # Triple: (head, relation, tail)
            gold_tri_set = set()
            for t in gold_tris_raw:
                gold_tri_set.add((
                    normalize_str(t.get("head")),
                    normalize_str(t.get("relation")),
                    normalize_str(t.get("tail"))
                ))

            # --- 解析 Prediction ---
            pred_ent_set = set()
            pred_tri_set = set()

            if llm_output:
                try:
                    # 清洗并解析 JSON
                    clean_json = llm_output.replace("```json", "").replace("```", "").strip()
                    pred_data = json.loads(clean_json)

                    for e in pred_data.get("entities", []):
                        pred_ent_set.add((normalize_str(e.get("name")), normalize_str(e.get("type"))))

                    for t in pred_data.get("triples", []):
                        pred_tri_set.add((
                            normalize_str(t.get("head")),
                            normalize_str(t.get("relation")),
                            normalize_str(t.get("tail"))
                        ))
                except:
                    print(f"⚠️ JSON 解析失败 (Row {idx})")

            # --- 计算单条数据的指标 ---
            # 实体指标
            ep, er, ef1, etp, efp, efn = calculate_f1(gold_ent_set, pred_ent_set)
            # 三元组指标
            tp_p, tp_r, tp_f1, ttp, tfp, tfn = calculate_f1(gold_tri_set, pred_tri_set)

            results.append({
                "segment_id": row.get("segment_id"),
                "text": row.get("text")[:50] + "...",  # 只存前50个字符方便查看
                "Entity_P": ep, "Entity_R": er, "Entity_F1": ef1,
                "Triple_P": tp_p, "Triple_R": tp_r, "Triple_F1": tp_f1,
                "Gold_Ent_Count": len(gold_ent_set),
                "Pred_Ent_Count": len(pred_ent_set),
                "Gold_Tri_Count": len(gold_tri_set),
                "Pred_Tri_Count": len(pred_tri_set),
                "LLM_Output": llm_output  # 保存原始输出方便 debug
            })

    # 3. 汇总统计
    res_df = pd.DataFrame(results)

    print("\n" + "=" * 30)
    print("📊 评估结果摘要 (Macro Average)")
    print("=" * 30)
    print(f"测试样本数: {len(res_df)}")
    print("-" * 20)
    print(f"【实体抽取 (NER)】")
    print(f"  Precision : {res_df['Entity_P'].mean():.4f}")
    print(f"  Recall    : {res_df['Entity_R'].mean():.4f}")
    print(f"  F1 Score  : {res_df['Entity_F1'].mean():.4f}")
    print("-" * 20)
    print(f"【三元组抽取 (RE)】")
    print(f"  Precision : {res_df['Triple_P'].mean():.4f}")
    print(f"  Recall    : {res_df['Triple_R'].mean():.4f}")
    print(f"  F1 Score  : {res_df['Triple_F1'].mean():.4f}")
    print("=" * 30)

    # 4. 保存详细报告
    res_df.to_excel(OUTPUT_REPORT, index=False)
    print(f"\n✅ 详细评估报告已保存至: {OUTPUT_REPORT}")
    print("建议打开报告查看 'Entity_F1' 或 'Triple_F1' 较低的行，进行 Bad Case 分析。")


if __name__ == "__main__":
    main()