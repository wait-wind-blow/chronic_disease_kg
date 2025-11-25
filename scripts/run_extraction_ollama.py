import json
import time
import re
from pathlib import Path
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置 =================
MODEL_NAME = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434/api/chat"

# 并发数 (建议保持 3-4)
MAX_WORKERS = 3

BASE_DIR = Path(__file__).resolve().parent.parent
IN_FILE = BASE_DIR / "data" / "processed" / "pubmed_segments.csv"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENTITY_OUT = OUT_DIR / "entity_raw.csv"
TRIPLE_OUT = OUT_DIR / "triple_raw.csv"

# ================= Prompt 定义 (保持不变) =================
NER_SYSTEM_PROMPT = """
1.假如你是医学领域抽取大模型，对医学文本数据进行抽取，回答的所有信息必须来自文本。
2.请严格按照给出模板指定的格式输出JSON内容，不要添加任何额外的内容。
3.实体类型 (type) 仅限以下四类（请严格区分 Chemical 和 Gene）：
  - "Disease": 疾病、症状、并发症、临床表型。
  - "Chemical": 药物、化学物质、疗法、**内源性小分子代谢物、脂类、离子**（如：神经酰胺、胆固醇、葡萄糖）。
  - "Gene": **仅限生物大分子**（基因、蛋白质、酶、受体、抗体），**严禁包含**脂类或小分子代谢物。
  - "Factor": 风险因素、生活方式、人口统计学指标。
4.确保输出的JSON格式没有转义符号，若未提取到任何实体，则列表为空；保留实体的英文原名填入 "name" 字段。
5.请根据以上JSON数据，准确提取文本中出现的所有关键医学实体，修复任何可能的错误或遗漏。
"""

NER_USER_TEMPLATE = """
##预抽取模板##
{{"Medical_Entities": [{{"name": "xxx", "type": "xxx"}}]}}

##示范案例##
输入：Metformin treatment significantly reduced HbA1c and plasma ceramides in patients with T2DM. High LDL-cholesterol is a risk factor.
输出：{{"Medical_Entities": [{{"name": "Metformin", "type": "Chemical"}}, {{"name": "HbA1c", "type": "Gene"}}, {{"name": "plasma ceramides", "type": "Chemical"}}, {{"name": "T2DM", "type": "Disease"}}, {{"name": "High LDL-cholesterol", "type": "Factor"}}]}}

##待抽取文本##
输入：
{TEXT}

输出：
"""

RE_SYSTEM_PROMPT = """
1.假如你是医学领域知识图谱构建大模型；对输入的 JSON 实体列表 和 原文文本 进行深度分析。
2.任务是构建实体间的语义关系（三元组），请严格按照“三元组模板”指定的格式输出内容。
3.关系类型 (relation) 仅限以下七类：
  - "Treat": 治疗/缓解 (Drug -> Disease)
  - "Prevent": 预防 (Chemical/Factor -> Disease)
  - "Cause": 导致/增加风险/副作用 (Factor/Chemical -> Disease)
  - "Complicate": 并发 (Disease -> Disease)
  - "Positive_associated": 正相关/指标升高/作为生物标志物 (A increases B, or A is high in B)
  - "Negative_associated": 负相关/指标降低/抑制 (A reduces B, or A is low in B)
  - "Associated": 其他相关/关系不明确
4.确保输出的三元组具有清晰的语义，仅在“输入1”提供的实体范围内构建关系，“head”和“tail”必须严格匹配输入 JSON 中的 "name"。
5.参考文本上下文，补充任何遗漏或修复可能出现的错误。
6.输出必须是合法的 JSON 列表，不要包含任何 Markdown 标记或其他无关内容，并且不需要换行。
"""

RE_USER_TEMPLATE = """
##三元组模板##
[{{"head": "实体A名称", "head_type": "实体A类型", "relation": "关系名称", "tail": "实体B名称", "tail_type": "实体B类型"}}]

##示范案例##
输入：
1. {{"Medical_Entities": [{{"name": "Metformin", "type": "Chemical"}}, {{"name": "T2DM", "type": "Disease"}}, {{"name": "lactic acidosis", "type": "Disease"}}, {{"name": "HbA1c", "type": "Gene"}}]}}
2. Metformin is widely used to treat T2DM but is associated with a risk of lactic acidosis. It effectively lowers HbA1c.
输出：
[{{"head": "Metformin", "head_type": "Chemical", "relation": "Treat", "tail": "T2DM", "tail_type": "Disease"}}, {{"head": "Metformin", "head_type": "Chemical", "relation": "Cause", "tail": "lactic acidosis", "tail_type": "Disease"}}, {{"head": "Metformin", "head_type": "Chemical", "relation": "Negative_associated", "tail": "HbA1c", "tail_type": "Gene"}}]

##待分析数据##
输入：
1. {ENTITIES_JSON}
2. {TEXT}

输出：
"""


# ================= 核心逻辑 (已增强 JSON 解析能力) =================

def call_ollama_api(system_prompt, user_prompt):
    payload = {
        "model": MODEL_NAME,
        "format": "json",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 4096}
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        if resp.status_code == 200:
            return resp.json()["message"]["content"]
    except Exception as e:
        print(f"API Error: {e}")
    return None


def extract_json_list(text):
    """
    终极暴力正则提取：兼容 [...]、{...} 以及 {"triples": [...]} 等各种包裹格式
    """
    if not text: return []

    # 预处理：去除 markdown 标记
    clean_text = text.replace("```json", "").replace("```", "").strip()

    # 定义一个内部函数来尝试“拆包”字典
    def unwrap_dict(d):
        # 常见的包裹 Key，模型喜欢用这些
        wrapper_keys = ["triples", "relationships", "relations", "results", "data", "output"]
        for k in wrapper_keys:
            if k in d and isinstance(d[k], list):
                return d[k]
        # 如果没有 wrapper key，但它自己长得像个三元组 (有 head/relation)，就返回 [d]
        if "head" in d and "relation" in d:
            return [d]
        return []

    results = []

    # 策略 1: 尝试直接解析整个字符串
    try:
        data = json.loads(clean_text)
        if isinstance(data, list):
            results = data
        elif isinstance(data, dict):
            results = unwrap_dict(data)
    except:
        pass

    if results: return results

    # 策略 2: 正则提取 [...] 列表
    try:
        match = re.search(r'\[.*\]', clean_text, re.DOTALL)
        if match:
            results = json.loads(match.group())
    except:
        pass

    if results: return results

    # 策略 3: 正则提取所有独立的 {...} 对象
    # 这种是为了应对模型输出了多个对象但忘记加逗号或方括号的情况
    try:
        # 匹配非嵌套的最外层大括号
        matches = re.findall(r'\{[^{}]+\}', clean_text)
        for m in matches:
            try:
                d = json.loads(m)
                if "head" in d and "relation" in d:
                    results.append(d)
            except:
                pass
    except:
        pass

    return results


def process_single_segment(row):
    segment_id = row["segment_id"]
    text = row["text"]
    pmid = row["pmid"]
    disease_group = row.get("disease_group", "")

    # --- Step 1: NER ---
    ner_output = call_ollama_api(NER_SYSTEM_PROMPT, NER_USER_TEMPLATE.format(TEXT=text))

    entities_list = []
    if ner_output:
        try:
            # 同样使用宽松解析
            clean_text = ner_output.replace("```json", "").replace("```", "").strip()
            # 尝试匹配 { ... }
            match = re.search(r'\{.*\}', clean_text, re.DOTALL)
            if match:
                ner_data = json.loads(match.group())
                # 兼容 Medical_Entities 或 entities
                raw_ents = ner_data.get("Medical_Entities") or ner_data.get("entities") or []

                for ent in raw_ents:
                    if isinstance(ent, dict) and "name" in ent and "type" in ent:
                        entities_list.append({
                            "segment_id": segment_id,
                            "pmid": pmid,
                            "disease_group": disease_group,
                            "entity_name": ent["name"],
                            "entity_type": ent["type"]
                        })
        except:
            pass

    if not entities_list:
        return [], []

    # --- Step 2: RE ---
    entities_json_str = json.dumps(
        {"Medical_Entities": [{"name": e["entity_name"], "type": e["entity_type"]} for e in entities_list]},
        ensure_ascii=False)

    re_output = call_ollama_api(RE_SYSTEM_PROMPT, RE_USER_TEMPLATE.format(ENTITIES_JSON=entities_json_str, TEXT=text))

    triples_list = []
    if re_output:
        # 使用增强版提取函数
        raw_triples = extract_json_list(re_output)

        if not raw_triples:
            # 如果还是失败，开启 Debug 打印
            # print(f"\n[Debug RE Fail] Segment: {segment_id}")
            # print(f"LLM Output: {re_output}...")
            pass

        for tri in raw_triples:
            if isinstance(tri, dict) and "head" in tri and "tail" in tri and "relation" in tri:
                triples_list.append({
                    "segment_id": segment_id,
                    "pmid": pmid,
                    "disease_group": disease_group,
                    "head_text": tri["head"],
                    "head_type": tri.get("head_type", ""),
                    "relation": tri["relation"],
                    "tail_text": tri["tail"],
                    "tail_type": tri.get("tail_type", ""),
                    "confidence": 1.0
                })

    return entities_list, triples_list


def main():
    if not IN_FILE.exists():
        print("Please run preprocess_pubmed.py first!")
        return

    df = pd.read_csv(IN_FILE)

    # ========= 测试开关 =========
    df = df.head(20)
    # ===========================

    print(f"Total segments to process: {len(df)}")

    all_entities = []
    all_triples = []

    print(f"Starting 2-Step Extraction with {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_row = {executor.submit(process_single_segment, row): row for _, row in df.iterrows()}

        for future in tqdm(as_completed(future_to_row), total=len(future_to_row)):
            ents, tris = future.result()
            all_entities.extend(ents)
            all_triples.extend(tris)

    # 3. 保存结果
    if all_entities:
        pd.DataFrame(all_entities).to_csv(ENTITY_OUT, index=False)
        print(f"\nEntities saved to {ENTITY_OUT} (Count: {len(all_entities)})")

    if all_triples:
        pd.DataFrame(all_triples).to_csv(TRIPLE_OUT, index=False)
        print(f"Triples saved to {TRIPLE_OUT} (Count: {len(all_triples)})")
    else:
        print("\n⚠️ Warning: No triples extracted.")


if __name__ == "__main__":
    main()