import pandas as pd
from pathlib import Path
import re
import html

# ========== 配置 ==========
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "pubmed"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "pubmed_segments.csv"


def clean_text(text: str) -> str:
    """
    学术级文本清洗函数：
    1. 转义 HTML 字符 (如 &gt; -> >)
    2. 去除 HTML 标签 (如 <i>, <b>, <sub>)
    3. 去除 URL 链接
    4. 规范化空白字符
    """
    if not isinstance(text, str):
        return ""

    # 1. HTML 解码
    text = html.unescape(text)

    # 2. 去除 HTML 标签 (保留标签内的内容，只去标签本身)
    text = re.sub(r'<[^>]+>', '', text)

    # 3. 去除 URL (http/https 开头)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # 4. 替换各种奇怪的空白符为单个空格
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def main():
    print("🚀 开始执行数据预处理...")
    all_rows = []

    # 1. 遍历所有 raw 数据 (pubmed_dm_cvd_5y.csv)
    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        print("❌ 未找到原始数据，请先运行 fetch_pubmed.py")
        return

    for csv_path in csv_files:
        print(f"正在处理文件: {csv_path.name} ...")
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"⚠️ 读取失败 {csv_path}: {e}")
            continue

        # 统计处理前数量
        original_count = len(df)

        # 2. 核心清洗逻辑
        # 填充空值
        df["title"] = df["title"].fillna("")
        df["abstract"] = df["abstract"].fillna("")

        # 应用清洗函数
        df["title_clean"] = df["title"].apply(clean_text)
        df["abstract_clean"] = df["abstract"].apply(clean_text)

        # 3. 过滤无效数据
        # 规则：摘要长度必须 > 50 字符，且标题不为空
        df = df[(df["abstract_clean"].str.len() > 50) & (df["title_clean"].str.len() > 5)]

        print(f"  - 清洗前: {original_count} 条 -> 清洗后: {len(df)} 条")

        # 4. 格式化输出
        for _, row in df.iterrows():
            pmid = str(row["pmid"])
            # 组合文本：Title. Abstract
            full_text = f"{row['title_clean']}. {row['abstract_clean']}"

            all_rows.append({
                "segment_id": f"pub_{pmid}",  # 唯一标识符
                "pmid": pmid,
                "year": row.get("year", ""),
                "disease_group": row.get("disease_group", "chronic"),
                "text": full_text,
                "source": "pubmed"
            })

    # 5. 保存结果
    if all_rows:
        result_df = pd.DataFrame(all_rows)
        # 按 pmid 去重（防止多次抓取导致的重复）
        result_df.drop_duplicates(subset=["pmid"], inplace=True)

        result_df.to_csv(OUT_FILE, index=False)
        print(f"\n✅ 预处理完成！")
        print(f"   - 总有效数据量: {len(result_df)} 条")
        print(f"   - 结果已保存至: {OUT_FILE}")

        # 打印一条样例，方便你检查质量
        print("\n📝 样例数据 (前 100 字符):")
        print(result_df.iloc[0]["text"][:100] + "...")
    else:
        print("\n⚠️ 没有生成任何有效数据，请检查原始 CSV 文件。")


if __name__ == "__main__":
    main()