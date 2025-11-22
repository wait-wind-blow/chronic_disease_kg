import pandas as pd
from pathlib import Path

# ========== 路径配置 ==========
BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"

ENTITY_RAW_FILE = PROC_DIR / "entity_raw.csv"
TRIPLE_RAW_FILE = PROC_DIR / "triple_raw.csv"

ENTITY_OUT_FILE = PROC_DIR / "entity_clean.csv"   # 对应 Neo4j 的 entity.csv
TRIPLE_OUT_FILE = PROC_DIR / "triple_clean.csv"   # 对应 Neo4j 的 triple.csv

# 允许的实体类型和关系类型
ALLOWED_ENTITY_TYPES = {"Disease", "Chemical", "Gene", "Factor"}
ALLOWED_RELATIONS = {
    "Treat",
    "Prevent",
    "Cause",
    "Complicate",
    "Associated",
    "Positive_associated",
    "Negative_associated",
}


def load_raw_data():
    print(f"Loading entity_raw from: {ENTITY_RAW_FILE}")
    ent_df = pd.read_csv(ENTITY_RAW_FILE)

    print(f"Loading triple_raw from: {TRIPLE_RAW_FILE}")
    tri_df = pd.read_csv(TRIPLE_RAW_FILE)

    print(f"Loaded {len(ent_df)} raw entities, {len(tri_df)} raw triples.")
    return ent_df, tri_df


def clean_entities(ent_df: pd.DataFrame):
    # 1. 基础清洗：去掉空 name、非法类型
    print("\n=== Cleaning entities ===")
    before = len(ent_df)
    ent_df["entity_name"] = ent_df["entity_name"].astype(str).str.strip()
    ent_df["entity_type"] = ent_df["entity_type"].astype(str).str.strip()

    ent_df = ent_df[ent_df["entity_name"] != ""]
    ent_df = ent_df[ent_df["entity_type"].isin(ALLOWED_ENTITY_TYPES)]
    after_basic = len(ent_df)
    print(f"After basic filter (non-empty name & allowed types): {after_basic} / {before}")

    # 2. 去重 (segment_id, pmid, entity_name, entity_type) 层面，避免同一段重复
    ent_df = ent_df.drop_duplicates(
        subset=["segment_id", "pmid", "entity_name", "entity_type"]
    )
    after_dedup_occ = len(ent_df)
    print(f"After dedup on occurrences: {after_dedup_occ}")

    # 3. 全局唯一实体：按 (entity_name, entity_type) 去重
    unique_ents = (
        ent_df[["entity_name", "entity_type"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    print(f"Unique (name,type) entities: {len(unique_ents)}")

    # 4. 分配实体 ID：按类型分段
    def assign_entity_ids(df_unique: pd.DataFrame) -> pd.DataFrame:
        ids = []
        counters = {"Disease": 1, "Chemical": 1, "Gene": 1, "Factor": 1}
        prefix = {"Disease": "D", "Chemical": "C", "Gene": "G", "Factor": "F"}
        for _, row in df_unique.iterrows():
            etype = row["entity_type"]
            idx = counters[etype]
            eid = f"{prefix[etype]}{idx:05d}"
            ids.append(eid)
            counters[etype] += 1
        df_unique["id"] = ids
        return df_unique

    unique_ents = assign_entity_ids(unique_ents)

    # 5. 构造最终 entity_clean.csv
    # Neo4j 导入需要的字段: id, name, zh_name, type, source
    entity_clean = unique_ents.copy()
    entity_clean = entity_clean.rename(
        columns={"entity_name": "name", "entity_type": "type"}
    )
    entity_clean["zh_name"] = ""          # 先留空，后续可以人工补中文名
    entity_clean["source"] = "pubmed"     # 来自 PubMed 抽取

    # 调整列顺序
    entity_clean = entity_clean[["id", "name", "zh_name", "type", "source"]]

    print(f"Final entity_clean rows: {len(entity_clean)}")
    return entity_clean, unique_ents


def build_entity_occurrence_mapping(ent_df: pd.DataFrame, unique_ents: pd.DataFrame):
    """
    为 triple 映射准备一个 (segment_id, pmid, entity_name) -> entity_id 的表。
    先将 ent_df 加上 entity_id，然后用于 triple 的 head/tail 对齐。
    """
    # 把全局 id 合并回每一条实体出现记录
    ent_with_id = ent_df.merge(
        unique_ents,
        on=["entity_name", "entity_type"],
        how="left",
        validate="m:1",
    )

    # 只保留必要的映射字段
    mapping = ent_with_id[["segment_id", "pmid", "entity_name", "id"]].copy()
    mapping = mapping.drop_duplicates()
    mapping = mapping.rename(columns={"entity_name": "name", "id": "entity_id"})

    print(f"Entity occurrence mapping rows: {len(mapping)}")
    return mapping


def clean_triples(tri_df: pd.DataFrame, ent_mapping: pd.DataFrame):
    print("\n=== Cleaning triples ===")
    before = len(tri_df)

    # 1. 基础清洗：关系合法 & 文本不空
    tri_df["relation"] = tri_df["relation"].astype(str).str.strip()
    tri_df["head_text"] = tri_df["head_text"].astype(str).str.strip()
    tri_df["tail_text"] = tri_df["tail_text"].astype(str).str.strip()

    tri_df = tri_df[tri_df["relation"].isin(ALLOWED_RELATIONS)]
    tri_df = tri_df[(tri_df["head_text"] != "") & (tri_df["tail_text"] != "")]
    after_basic = len(tri_df)
    print(f"After basic filter (allowed relations & non-empty head/tail): {after_basic} / {before}")

    # 2. 将 triple 的 head_text / tail_text 映射到 entity_id
    # 通过 (segment_id, pmid, name) 与 ent_mapping 连接
    head_map = ent_mapping.rename(
        columns={"name": "head_text", "entity_id": "head_id"}
    )
    tail_map = ent_mapping.rename(
        columns={"name": "tail_text", "entity_id": "tail_id"}
    )

    tri_df = tri_df.merge(
        head_map[["segment_id", "pmid", "head_text", "head_id"]],
        on=["segment_id", "pmid", "head_text"],
        how="left",
    )
    tri_df = tri_df.merge(
        tail_map[["segment_id", "pmid", "tail_text", "tail_id"]],
        on=["segment_id", "pmid", "tail_text"],
        how="left",
    )

    # 3. 丢弃找不到实体 ID 的三元组
    before_drop = len(tri_df)
    tri_df = tri_df.dropna(subset=["head_id", "tail_id"])
    after_drop = len(tri_df)
    print(f"After drop triples without head_id/tail_id: {after_drop} / {before_drop}")

    # 4. 聚合去重：按 (head_id, relation, tail_id) 合并，多文献可统计 freq
    # 如果你想保留 doc 信息，可以后续扩展；这里先做简单图谱版
    tri_df["confidence"] = pd.to_numeric(tri_df["confidence"], errors="coerce")
    grouped = tri_df.groupby(["head_id", "relation", "tail_id"], as_index=False).agg(
        freq=("pmid", "count"),
        confidence=("confidence", "max"),
    )

    print(f"Unique triples (by head_id, relation, tail_id): {len(grouped)}")

    # 5. 为每条 triple 分配 id
    def assign_triple_ids(df: pd.DataFrame) -> pd.DataFrame:
        ids = [f"R{i:05d}" for i in range(1, len(df) + 1)]
        df["id"] = ids
        return df

    grouped = assign_triple_ids(grouped)

    # 6. 构造最终 triple_clean.csv
    triple_clean = grouped.copy()
    triple_clean["source"] = "pubmed"

    # 调整列顺序：id, head_id, relation, tail_id, source, freq, confidence
    triple_clean = triple_clean[
        ["id", "head_id", "relation", "tail_id", "source", "freq", "confidence"]
    ]

    return triple_clean


def main():
    # 1. 读原始数据
    ent_df_raw, tri_df_raw = load_raw_data()

    # 2. 清洗实体，生成 entity_clean & 全局唯一实体表
    entity_clean, unique_ents = clean_entities(ent_df_raw)

    # 3. 构建 (segment_id, pmid, name) -> entity_id 映射
    ent_mapping = build_entity_occurrence_mapping(ent_df_raw, unique_ents)

    # 4. 清洗三元组，生成 triple_clean
    triple_clean = clean_triples(tri_df_raw, ent_mapping)

    # 5. 保存结果
    ENTITY_OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    entity_clean.to_csv(ENTITY_OUT_FILE, index=False)
    print(f"\nSaved entity_clean to: {ENTITY_OUT_FILE}")

    TRIPLE_OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    triple_clean.to_csv(TRIPLE_OUT_FILE, index=False)
    print(f"Saved triple_clean to: {TRIPLE_OUT_FILE}")


if __name__ == "__main__":
    main()
