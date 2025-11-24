import pandas as pd
from pathlib import Path

# ==== 路径配置 ====
BASE_DIR = Path(__file__).resolve().parent.parent
IN_FILE = BASE_DIR / "data" / "raw" / "pubmed" / "pubmed_chronic_5y.csv"
OUT_FILE = BASE_DIR / "data" / "processed" / "pubmed_chronic_5y_tagged.csv"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ==== 关键词字典：可以按需自己继续加 ====

CATEGORY_PATTERNS = {
    "diabetes": [
        "type 2 diabetes",
        "type ii diabetes",
        "t2dm",
        "t2d",
        "diabetes mellitus",
        "diabetes",
        "prediabetes",
        "pre-diabetes",
        "impaired glucose tolerance",
        "impaired fasting glucose",
    ],
    "cardiovascular": [
        "hypertension",
        "high blood pressure",
        "elevated blood pressure",
        "coronary artery disease",
        "coronary heart disease",
        "ischemic heart disease",
        "ischaemic heart disease",
        "myocardial infarction",
        "heart failure",
        "atrial fibrillation",
        "arrhythmia",
        "stroke",
        "cerebrovascular",
        "transient ischemic attack",
        "tia ",
        "peripheral artery disease",
        "peripheral arterial disease",
        "atherosclerosis",
        "cardiovascular disease",
        "cvd ",
    ],
    "respiratory": [
        "copd",
        "chronic obstructive pulmonary disease",
        "chronic obstructive lung disease",
        "asthma",
        "emphysema",
        "chronic bronchitis",
        "bronchiectasis",
        "chronic respiratory",
    ],
    "cancer": [
        "cancer",
        "carcinoma",
        "neoplasm",
        "neoplasia",
        "tumor",
        "tumour",
        "leukemia",
        "lymphoma",
        "melanoma",
        "sarcoma",
        "glioma",
        "myeloma",
    ],
    "ckd": [
        "chronic kidney disease",
        "ckd",
        "end-stage renal disease",
        "end stage renal disease",
        "esrd",
        "diabetic nephropathy",
        "kidney failure",
        "renal failure",
        "chronic renal insufficiency",
    ],
    "other_chronic": [
        "rheumatoid arthritis",
        "osteoarthritis",
        "osteoporosis",
        "alzheimer",
        "dementia",
        "parkinson",
        "chronic liver disease",
        "liver cirrhosis",
        "cirrhosis",
        "chronic hepatitis",
        "psoriasis",
        "inflammatory bowel disease",
        "crohn's disease",
        "ulcerative colitis",
        "multiple sclerosis",
        "epilepsy",
        "chronic pain",
    ],
}

# 主类别优先级（如果多标签，只选一个 primary_category 按这个顺序）
PRIMARY_PRIORITY = [
    "diabetes",
    "cardiovascular",
    "respiratory",
    "cancer",
    "ckd",
    "other_chronic",
]


def classify_text(text: str):
    """
    根据标题+摘要文本，返回一个 set，例如：
    {"diabetes", "cardiovascular"}
    """
    if not isinstance(text, str):
        return set()

    t = text.lower()
    categories = set()

    for cat, patterns in CATEGORY_PATTERNS.items():
        for p in patterns:
            if p.lower() in t:
                categories.add(cat)
                break  # 这个类别命中一个关键词就够了

    return categories


def choose_primary_category(categories: set):
    """
    从多标签中选一个主类别（primary_category），按 PRIORITY 顺序选第一个命中的。
    如果没有任何类别，就返回 "unknown"。
    """
    if not categories:
        return "unknown"

    for cat in PRIMARY_PRIORITY:
        if cat in categories:
            return cat

    # 理论上不会走到这一步，但以防万一
    return sorted(categories)[0]


def main():
    print(f"读取原始 CSV：{IN_FILE}")
    df = pd.read_csv(IN_FILE)

    # 合并标题和摘要作为文本
    print("开始按标题+摘要进行粗分类 ...")
    texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).astype(str)

    all_categories = []
    primary_categories = []

    for i, txt in enumerate(texts):
        cats = classify_text(txt)
        all_categories.append(",".join(sorted(cats)) if cats else "")
        primary_categories.appendchoose_primary_category(cats))

        if (i + 1) % 50000 == 0:
            print(f"  已处理 {i + 1} 条记录...")

    df["category_list"] = all_categories
    df["primary_category"] = primary_categories

    print(f"写出带标签的 CSV 到：{OUT_FILE}")
    df.to_csv(OUT_FILE, index=False)
    print("完成。")


if __name__ == "__main__":
    main()
