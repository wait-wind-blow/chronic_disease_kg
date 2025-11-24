import json
import time
import re
from pathlib import Path
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # 建议安装：pip install tqdm，用来显示进度条

# ================= 配置 =================
MODEL_NAME = "qwen3:8b"  # 确保你本地 ollama 已经 pull 了这个模型
OLLAMA_URL = "http://localhost:11434/api/chat"

# 并发数：根据你的显存决定。
# 如果是 24G 显存，可以试着开 5-8；如果是 12G，建议 2-4。
# 如果 Ollama 是串行处理的（没有配置 parallel），开多线程也能利用网络IO时间，设为 4 比较稳妥。
MAX_WORKERS = 4

BASE_DIR = Path(__file__).resolve().parent.parent
# 注意：这里读取的是 preprocess 生成的文件，确保 preprocess_pubmed.py 也运行过
IN_FILE = BASE_DIR / "data" / "processed" / "pubmed_segments.csv"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENTITY_OUT = OUT_DIR / "entity_raw.csv"
TRIPLE_OUT = OUT_DIR / "triple_raw.csv"

# ================= 升级版 Prompt (Few-shot) =================
SYSTEM_PROMPT = """
You are an expert medical knowledge graph assistant. Extract entities and relations from the text.

[Schema]
Entities: "Disease", "Chemical" (drugs), "Gene", "Factor" (risk/protective factors).
Relations: 
- "Treat": Drug/Factor -> Disease
- "Prevent": Drug/Factor -> Disease
- "Cause": Factor -> Disease
- "Complicate": Disease -> Disease (comorbidities)
- "Positive_associated" / "Negative_associated": Factor/Gene <-> Disease
- "Associated": General association

[Examples]
Input: "Metformin treatment significantly reduced HbA1c in T2DM patients. However, high BMI is a risk factor for heart failure."
Output:
{
  "entities": [
    {"name": "Metformin", "type": "Chemical"},
    {"name": "T2DM", "type": "Disease"},
    {"name": "BMI", "type": "Factor"},
    {"name": "heart failure", "type": "Disease"}
  ],
  "triples": [
    {"head": "Metformin", "relation": "Treat", "tail": "T2DM", "confidence": 0.95},
    {"head": "BMI", "relation": "Positive_associated", "tail": "heart failure", "confidence": 0.9}
  ]
}

[Requirements]
1. Output ONLY valid JSON. No markdown, no explanations.
2. Keep entity names in their original language (English).
3. If no entities found, return {"entities": [], "triples": []}.
"""

USER_TEMPLATE = "Text: {TEXT}"


# ================= 核心逻辑 =================

def call_ollama(text: str, segment_id: str):
    """发送请求给 Ollama"""
    payload = {
        "model": MODEL_NAME,
        "format": "json",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(TEXT=text)},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 2048}  # 低温度减少幻觉
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if resp.status_code == 200:
            return segment_id, resp.json()["message"]["content"], None
        return segment_id, None, f"Status {resp.status_code}"
    except Exception as e:
        return segment_id, None, str(e)


def parse_result(segment_id, llm_output, row_data):
    """解析 LLM 返回的 JSON"""
    entities_data = []
    triples_data = []

    try:
        # 尝试清洗 JSON（有些模型喜欢加 ```json ... ```）
        clean_json = llm_output.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)

        for ent in data.get("entities", []):
            entities_data.append({
                "segment_id": segment_id,
                "pmid": row_data["pmid"],
                "disease_group": row_data["disease_group"],
                "entity_name": ent.get("name"),
                "entity_type": ent.get("type")
            })

        for tri in data.get("triples", []):
            triples_data.append({
                "segment_id": segment_id,
                "pmid": row_data["pmid"],
                "disease_group": row_data["disease_group"],
                "head_text": tri.get("head"),
                "relation": tri.get("relation"),
                "tail_text": tri.get("tail"),
                "confidence": tri.get("confidence")
            })

    except Exception as e:
        # 解析失败时，记录一下（可选：写入 error log）
        # print(f"Parse error {segment_id}: {e}")
        pass

    return entities_data, triples_data


def main():
    # 1. 读取数据
    if not IN_FILE.exists():
        print("Please run preprocess_pubmed.py first!")
        return

    df = pd.read_csv(IN_FILE)
    # 测试阶段只跑前 50 条，正式跑时去掉 .head()
    df = df.head(50)
    print(f"Total segments to process: {len(df)}")

    # 2. 并发处理
    all_entities = []
    all_triples = []

    # 建立 segment_id 到 row 的映射，方便后续查找
    rows_map = {row["segment_id"]: row for _, row in df.iterrows()}

    print(f"Starting extraction with {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任务
        future_to_seg = {
            executor.submit(call_ollama, row["text"], row["segment_id"]): row["segment_id"]
            for _, row in df.iterrows()
        }

        # 使用 tqdm 显示进度条
        for future in tqdm(as_completed(future_to_seg), total=len(future_to_seg)):
            seg_id, content, error = future.result()

            if content:
                ents, tris = parse_result(seg_id, content, rows_map[seg_id])
                all_entities.extend(ents)
                all_triples.extend(tris)
            else:
                # 如果报错，可以选择重试或记录日志
                pass

    # 3. 保存结果
    if all_entities:
        pd.DataFrame(all_entities).to_csv(ENTITY_OUT, index=False)
        print(f"Entities saved to {ENTITY_OUT}")

    if all_triples:
        pd.DataFrame(all_triples).to_csv(TRIPLE_OUT, index=False)
        print(f"Triples saved to {TRIPLE_OUT}")


if __name__ == "__main__":
    main()