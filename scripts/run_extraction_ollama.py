import json
import time
import re
from pathlib import Path

import pandas as pd
import requests

# ================= 基本配置 =================

# 如果 qwen3:8b 太慢，可以改成更小的模型，例如 "qwen2:4b"
MODEL_NAME = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434/api/chat"

BASE_DIR = Path(__file__).resolve().parent.parent
IN_FILE = BASE_DIR / "data" / "processed" / "pubmed_segments.csv"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENTITY_OUT = OUT_DIR / "entity_raw.csv"
TRIPLE_OUT = OUT_DIR / "triple_raw.csv"

# 一次先跑多少条测试
MAX_SEGMENTS = 30

# 允许的关系类型集合
ALLOWED_RELATIONS = {
    "Treat",
    "Prevent",
    "Cause",
    "Complicate",
    "Associated",
    "Positive_associated",
    "Negative_associated",
}

# ================= Prompt 定义 =================

SYSTEM_PROMPT = """
You are an assistant for building a medical knowledge graph. 
Your task is to extract entities and triples from medical texts (English abstracts or Chinese guidelines).

[Entity types] (must be one of the following 4 strings)
- "Disease": diseases or syndromes, including chronic diseases, complications, outcomes
- "Chemical": drugs or chemical substances
- "Gene": genes, proteins, receptors
- "Factor": risk or protective factors, including lifestyle, diet, nutrients, behaviors, and outcome indicators

[Relation types] (must be one of the following 7 strings)
- "Treat": used to treat a disease
- "Prevent": used to prevent a disease or its complications
- "Cause": causes or significantly increases the risk of a disease/adverse outcome
- "Complicate": complication/comorbidity relationships between diseases
- "Associated": associated, but direction and sign are unclear
- "Positive_associated": positive association, one increases when the other increases
- "Negative_associated": negative association, one decreases when the other increases

[Language rules]
- If the input text is in English, KEEP ALL ENTITY NAMES IN ENGLISH. Do NOT translate them.
- If the input text is in Chinese, use Chinese entity names.
- Do NOT change the language of terms.

[STRICT FORMAT REQUIREMENT]
- The field "type" of each entity MUST be exactly one of:
  "Disease", "Chemical", "Gene", "Factor"
- The field "relation" of each triple MUST be exactly one of:
  "Treat", "Prevent", "Cause", "Complicate", 
  "Associated", "Positive_associated", "Negative_associated"
- Do NOT invent new relation names.
- If you are not sure which relation type to use, prefer "Associated".

[OUTPUT FORMAT]
You MUST output a SINGLE JSON object with the following structure, and NOTHING ELSE:

{
  "entities": [
    {"name": "entity name 1", "type": "Disease"},
    {"name": "entity name 2", "type": "Factor"}
  ],
  "triples": [
    {
      "head": "head entity name",
      "relation": "RelationType",
      "tail": "tail entity name",
      "confidence": 0.0
    }
  ]
}

- "confidence" MUST be a number between 0.0 and 1.0.
- If you cannot find any valid entities or triples, output:
{"entities": [], "triples": []}
"""

USER_TEMPLATE = """
Below is a medical text related to chronic diseases (usually an English abstract).
Please extract entities and triples according to the definitions.

Text:
\"\"\" 
{TEXT}
\"\"\"

Remember:
- Output ONLY a single JSON object.
- Do NOT output explanations, comments, or any extra text.
"""


# ================= 工具函数 =================

def normalize_relation(rel: str) -> str | None:
    """
    对关系类型字符串做简单清洗和纠错。
    如果无法映射到允许集合，返回 None。
    """
    if not isinstance(rel, str):
        return None
    rel = rel.strip()

    # 完全匹配
    if rel in ALLOWED_RELATIONS:
        return rel

    # 一些常见小错误的纠偏
    lower = rel.lower()
    if "positive" in lower and "associ" in lower:
        return "Positive_associated"
    if "negative" in lower and "associ" in lower:
        return "Negative_associated"
    if "complica" in lower:
        return "Complicate"
    if "prevent" in lower:
        return "Prevent"
    if "treat" in lower:
        return "Treat"
    if "cause" in lower:
        return "Cause"
    if "associ" in lower:
        return "Associated"

    # 其余情况丢弃
    return None


def clean_json_string(text: str) -> str:
    """
    尝试把“差一点合法”的 JSON 字符串修一修：
    - 截取第一个 { 到 最后一个 }
    - 去掉 ]} 前面的尾逗号
    - 把 True/False/None 换成 true/false/null
    """
    if not isinstance(text, str):
        return ""

    # 截取第一个 { 到最后一个 }
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        text = text[start:end]
    except ValueError:
        # 找不到大括号，就直接返回原文
        return text

    # 去掉列表或对象最后一个元素后的尾逗号
    # 例如 "[1,2,]" -> "[1,2]"， "{... ,}" -> "{...}"
    text = re.sub(r",(\s*[\]}])", r"\1", text)

    # 替换 Python 风格的 True/False/None
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)

    return text


def parse_with_regex(text: str) -> dict:
    """
    解析失败时的兜底方案：
    直接用正则在文本中抓取形如
      {"name": "...", "type": "..."}
      {"head": "...", "relation": "...", "tail": "...", "confidence": ...}
    的小对象，忽略不完整的残片。
    """
    entities = []
    triples = []

    # 匹配实体对象
    ent_pattern = re.compile(
        r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"type"\s*:\s*"([^"]+)"\s*\}'
    )
    for name, etype in ent_pattern.findall(text):
        entities.append({"name": name, "type": etype})

    # 匹配三元组对象（confidence 是数字）
    tri_pattern = re.compile(
        r'\{\s*"head"\s*:\s*"([^"]+)"\s*,\s*"relation"\s*:\s*"([^"]+)"\s*,\s*"tail"\s*:\s*"([^"]+)"\s*,\s*"confidence"\s*:\s*([0-9.]+)'
    )
    for head, rel, tail, conf_str in tri_pattern.findall(text):
        try:
            conf = float(conf_str)
        except ValueError:
            conf = None
        triples.append(
            {
                "head": head,
                "relation": rel,
                "tail": tail,
                "confidence": conf,
            }
        )

    return {"entities": entities, "triples": triples}


def extract_json_from_text(text: str):
    """
    尝试从模型返回的字符串中解析出结构化结果。
    1) 优先 json.loads
    2) 失败则 clean_json_string + json.loads
    3) 再失败则用正则抽取实体和三元组
    """
    if not isinstance(text, str):
        return None
    raw = text.strip()

    # 1) 直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2) 清洗后解析
    cleaned = clean_json_string(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3) 兜底：正则抽取
    fallback = parse_with_regex(raw)
    if fallback["entities"] or fallback["triples"]:
        return fallback

    return None


def call_ollama(text: str) -> str:
    """调用本地 Ollama 模型，返回模型输出的 content 字符串。"""
    user_prompt = USER_TEMPLATE.format(TEXT=text)

    payload = {
        "model": MODEL_NAME,
        "format": "json",  # 尽量让 Ollama 返回 JSON
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "num_predict": 256,   # 限制生成长度
            "num_ctx": 2048,
            "temperature": 0.2,
        },
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    content = data["message"]["content"]
    return content


# ================= 主流程 =================

def main():
    # 1. 读取预处理好的 segments
    df = pd.read_csv(IN_FILE)
    print(f"Loaded {len(df)} segments from {IN_FILE}")

    # 只取前 MAX_SEGMENTS 条测试
    df = df.head(MAX_SEGMENTS)

    entity_rows = []
    triple_rows = []

    for idx, row in df.iterrows():
        segment_id = row["segment_id"]
        pmid = str(row["pmid"])
        disease_group = row.get("disease_group", "")
        text = str(row["text"])

        print(f"\n=== [{idx + 1}/{len(df)}] segment_id={segment_id}, pmid={pmid} ===")

        try:
            llm_output = call_ollama(text)
            preview = llm_output[:200].replace("\n", " ")
            print("LLM raw output (truncated):", preview, "...")
        except Exception as e:
            print(f"Error calling Ollama for segment {segment_id}: {e}")
            continue

        data = extract_json_from_text(llm_output)
        if data is None:
            print(f"Failed to parse JSON/regex for segment {segment_id}")
            # 把原始输出保存下来，方便你手动查看具体哪里有问题
            debug_file = OUT_DIR / f"debug_{segment_id}.txt"
            try:
                debug_file.write_text(llm_output, encoding="utf-8")
                print(f"Raw LLM output saved to {debug_file}")
            except Exception as e:
                print(f"Failed to save debug file for {segment_id}: {e}")
            continue

        entities = data.get("entities", []) or []
        triples = data.get("triples", []) or []

        # 收集实体
        for ent in entities:
            name = (ent.get("name") or "").strip()
            etype = (ent.get("type") or "").strip()
            if not name:
                continue
            entity_rows.append(
                {
                    "segment_id": segment_id,
                    "pmid": pmid,
                    "disease_group": disease_group,
                    "entity_name": name,
                    "entity_type": etype,
                    "model": MODEL_NAME,
                    "source": "pubmed",
                }
            )

        # 收集三元组
        for tri in triples:
            head = (tri.get("head") or "").strip()
            relation_raw = (tri.get("relation") or "").strip()
            tail = (tri.get("tail") or "").strip()
            conf = tri.get("confidence", None)

            relation = normalize_relation(relation_raw)
            if not head or not tail or not relation:
                continue

            triple_rows.append(
                {
                    "segment_id": segment_id,
                    "pmid": pmid,
                    "disease_group": disease_group,
                    "head_text": head,
                    "relation": relation,
                    "tail_text": tail,
                    "confidence": conf,
                    "model": MODEL_NAME,
                    "source": "pubmed",
                }
            )

        # 如果你换成小模型、速度还可以，可以把下一行注释掉
        time.sleep(1)

    # 保存结果
    if entity_rows:
        ent_df = pd.DataFrame(entity_rows)
        ent_df.to_csv(ENTITY_OUT, index=False)
        print(f"\nSaved {len(ent_df)} entities to {ENTITY_OUT}")
    else:
        print("\nNo entities extracted.")

    if triple_rows:
        tri_df = pd.DataFrame(triple_rows)
        tri_df.to_csv(TRIPLE_OUT, index=False)
        print(f"Saved {len(tri_df)} triples to {TRIPLE_OUT}")
    else:
        print("No triples extracted.")


if __name__ == "__main__":
    main()
