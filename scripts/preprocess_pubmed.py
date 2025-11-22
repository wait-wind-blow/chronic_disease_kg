import pandas as pd
from pathlib import Path
import re

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw" / "pubmed"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """基础清洗：去掉换行、多余空格"""
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    all_rows = []

    # 遍历 pubmed_*.csv
    for csv_path in RAW_DIR.glob("pubmed_*.csv"):
        print(f"Processing {csv_path.name} ...")
        df = pd.read_csv(csv_path)

        # 清洗 title 和 abstract
        df["title"] = df["title"].apply(clean_text)
        df["abstract"] = df["abstract"].apply(clean_text)

        # 丢掉没有摘要或太短的摘要（长度 <= 50 的）
        df = df[df["abstract"].str.len() > 50]

        # disease_group 从文件里读取（我们之前保存过）
        if "disease_group" in df.columns and len(df) > 0:
            group = df["disease_group"].iloc[0]
        else:
            # 兜底：从文件名里截
            group = csv_path.stem.split("_", 1)[1]

        for _, row in df.iterrows():
            pmid = str(row["pmid"])
            title = row["title"]
            abstract = row["abstract"]
            year = row.get("year", None)

            # 合成一个文本片段：标题 + 摘要
            if abstract:
                text = f"{title}. {abstract}"
            else:
                text = title

            all_rows.append(
                {
                    "segment_id": f"{group}_{pmid}",  # 片段ID：方便追踪
                    "pmid": pmid,
                    "disease_group": group,
                    "year": year,
                    "text": text,
                    "source": "pubmed",
                }
            )

    segments_df = pd.DataFrame(all_rows)

    # 防止重复：按 pmid 去重
    segments_df = segments_df.drop_duplicates(subset=["pmid"])

    out_file = OUT_DIR / "pubmed_segments.csv"
    segments_df.to_csv(out_file, index=False)
    print(f"Saved {len(segments_df)} segments to {out_file}")


if __name__ == "__main__":
    main()
