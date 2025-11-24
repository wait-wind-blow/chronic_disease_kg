import pandas as pd
from pathlib import Path

# ========== 配置 ==========
BASE_DIR = Path(__file__).resolve().parent.parent
IN_FILE = BASE_DIR / "data" / "processed" / "pubmed_segments.csv"
OUT_DIR = BASE_DIR / "data" / "gold_standard"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "gold_standard_to_annotate.xlsx"  # 保存为 Excel 方便人工编辑
SAMPLE_SIZE = 50  # 抽样数量


def main():
    if not IN_FILE.exists():
        print(f"❌ 未找到输入文件: {IN_FILE}，请先运行 preprocess 脚本。")
        return

    print(f"正在读取数据: {IN_FILE} ...")
    df = pd.read_csv(IN_FILE)

    if len(df) < SAMPLE_SIZE:
        print(f"⚠️ 数据量不足 {SAMPLE_SIZE} 条，将使用全部数据。")
        sample_df = df
    else:
        # 随机抽样，设置 random_state 保证每次抽的一样（方便复现）
        sample_df = df.sample(n=SAMPLE_SIZE, random_state=42)

    print(f"已随机抽取 {len(sample_df)} 条数据。")

    # 创建适合人工标注的列结构
    # 我们保留原文，留空列让你填
    annotation_df = sample_df[["segment_id", "pmid", "text"]].copy()

    # 增加标注列（人工填写）
    annotation_df["human_entities"] = ""  # 格式示例: Metformin|Chemical; T2DM|Disease
    annotation_df["human_triples"] = ""  # 格式示例: Metformin|Treat|T2DM
    annotation_df["notes"] = ""  # 备注

    # 保存
    try:
        annotation_df.to_excel(OUT_FILE, index=False)
        print(f"\n✅ 金标准采样完成！")
        print(f"   - 请打开文件: {OUT_FILE}")
        print(f"   - 任务: 请在 'human_entities' 和 'human_triples' 列中手动填写正确的抽取结果。")
        print(f"   - 这将作为计算 F1 分数的依据。")
    except ImportError:
        print("❌ 保存 Excel 失败，请确保安装了 openpyxl 库 (pip install openpyxl)")
        # 降级保存为 CSV
        csv_file = OUT_FILE.with_suffix(".csv")
        annotation_df.to_csv(csv_file, index=False)
        print(f"   - 已降级保存为 CSV: {csv_file}")


if __name__ == "__main__":
    main()