import pandas as pd
from pathlib import Path

# ========== 路径配置 ==========
BASE_DIR = Path(__file__).resolve().parent.parent

# PubMed 原始 CSV 存放目录（前面我们就是放这里）
RAW_PUBMED_DIR = BASE_DIR / "data" / "raw" / "pubmed"

# 文献信息 Excel 模板路径（建议你把下载的那个文件放到 data/meta 下）
META_DIR = BASE_DIR / "data" / "meta"
META_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_PATH = META_DIR / "慢性病文献信息管理模板.xlsx"

# literature_meta 表的列（必须和 Excel 里一致）
COLUMNS = [
    "literature_id",
    "title_zh",
    "title_en",
    "first_author",
    "year",
    "journal_or_publisher",
    "country_or_region",
    "language",
    "disease_system",
    "disease_category",
    "disease_stage_or_risk",
    "document_type",
    "evidence_level",
    "source_db",
    "source_url",
    "local_file_path",
    "fulltext_available",
    "text_extracted",
    "included_in_goldset",
    "annotation_status",
    "used_in_training",
    "used_in_dev",
    "used_in_test",
    "key_topics",
    "notes",
]


def map_disease_group(group: str):
    """根据我们之前的 disease_group 映射到疾病系统和疾病类别"""
    group = (group or "").lower()
    if group == "diabetes":
        return "代谢", "2型糖尿病/糖尿病前期"
    elif group == "cardiovascular":
        return "心血管", "心血管疾病"
    elif group == "respiratory":
        return "呼吸", "慢阻肺/哮喘"
    elif group == "cancer":
        return "肿瘤", "结直肠癌/乳腺癌等"
    else:
        return "", ""


def load_existing_meta():
    """读取已有的文献信息 Excel，如果没有就新建空表"""
    if not EXCEL_PATH.exists():
        print(f"Excel 文件不存在，将创建新的：{EXCEL_PATH}")
        meta_df = pd.DataFrame(columns=COLUMNS)
        dict_df = pd.DataFrame({"字段名": COLUMNS, "说明": ""})  # 占位，防止报错
        return meta_df, dict_df

    # 读取已有 Excel
    xls = pd.ExcelFile(EXCEL_PATH)
    if "literature_meta" in xls.sheet_names:
        meta_df = pd.read_excel(EXCEL_PATH, sheet_name="literature_meta")
    else:
        meta_df = pd.DataFrame(columns=COLUMNS)

    if "data_dict" in xls.sheet_names:
        dict_df = pd.read_excel(EXCEL_PATH, sheet_name="data_dict")
    else:
        dict_df = pd.DataFrame({"字段名": COLUMNS, "说明": ""})

    # 确保列齐全
    for col in COLUMNS:
        if col not in meta_df.columns:
            meta_df[col] = ""

    meta_df = meta_df[COLUMNS]

    print(f"Loaded existing literature_meta rows: {len(meta_df)}")
    return meta_df, dict_df


def build_new_rows_from_pubmed():
    """从 pubmed_*.csv 构建需要插入 Excel 的新行"""
    all_rows = []

    for csv_path in RAW_PUBMED_DIR.glob("pubmed_*.csv"):
        print(f"Processing {csv_path.name} ...")
        df = pd.read_csv(csv_path)

        # 确保列存在
        for col in ["pmid", "title", "year", "disease_group"]:
            if col not in df.columns:
                raise ValueError(f"{csv_path} 缺少必要列: {col}")

        # 遍历每一篇文献
        for _, row in df.iterrows():
            pmid = str(row["pmid"])
            title = str(row["title"])
            year = row.get("year", None)
            disease_group = row.get("disease_group", "")

            disease_system, disease_category = map_disease_group(disease_group)

            literature_id = f"PMID_{pmid}"
            source_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            new_row = {
                "literature_id": literature_id,
                "title_zh": "",
                "title_en": title,
                "first_author": "",
                "year": year,
                "journal_or_publisher": "",
                "country_or_region": "",
                "language": "English",
                "disease_system": disease_system,
                "disease_category": disease_category,
                "disease_stage_or_risk": "",
                "document_type": "原始研究/综述（待确认）",
                "evidence_level": "",
                "source_db": "PubMed",
                "source_url": source_url,
                "local_file_path": "",
                "fulltext_available": "未知",
                "text_extracted": "是",  # 因为我们已经有 csv/segments
                "included_in_goldset": "",
                "annotation_status": "",
                "used_in_training": "",
                "used_in_dev": "",
                "used_in_test": "",
                "key_topics": "",
                "notes": "",
            }

            all_rows.append(new_row)

    new_df = pd.DataFrame(all_rows, columns=COLUMNS)
    print(f"Built {len(new_df)} rows from PubMed csvs.")
    return new_df


def merge_and_save(meta_df: pd.DataFrame, dict_df: pd.DataFrame, new_df: pd.DataFrame):
    """把新文献合并进已有 Excel，并写回文件"""
    if meta_df.empty:
        existing_urls = set()
    else:
        existing_urls = set(meta_df["source_url"].astype(str).tolist())

    # 只保留 Excel 里还没有的文献
    mask_new = ~new_df["source_url"].astype(str).isin(existing_urls)
    to_add = new_df[mask_new].copy()
    print(f"New rows to add (not in excel yet): {len(to_add)}")

    if len(to_add) == 0:
        print("没有新文献需要添加，退出。")
        return

    updated_meta = pd.concat([meta_df, to_add], ignore_index=True)

    # 写回 Excel（覆盖原文件）
    EXCEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl", mode="w") as writer:
        updated_meta.to_excel(writer, sheet_name="literature_meta", index=False)
        dict_df.to_excel(writer, sheet_name="data_dict", index=False)

    print(f"Saved updated excel to {EXCEL_PATH}")
    print(f"Total literature_meta rows: {len(updated_meta)}")


def main():
    meta_df, dict_df = load_existing_meta()
    new_df = build_new_rows_from_pubmed()
    merge_and_save(meta_df, dict_df, new_df)


if __name__ == "__main__":
    main()
