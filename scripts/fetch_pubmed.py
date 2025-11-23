from Bio import Entrez
from pathlib import Path
from datetime import date, timedelta
import csv
import time
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError
import socket


# ========== 1. PubMed 账号配置 ==========
Entrez.email = "YOUR_EMAIL_HERE"  # TODO: 换成你的邮箱
# 如果你有 NCBI API key，可以解开下面一行，让速度安全高一点
# Entrez.api_key = "YOUR_API_KEY_HERE"

# ========== 2. 路径配置 ==========
BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "data" / "raw" / "pubmed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "pubmed_chronic_5y.csv"
INTERVALS_FILE = OUT_DIR / "pubmed_chronic_5y_intervals.json"


# ========== 3. 检索参数 ==========
QUERY_MESH = '"Chronic Disease"[Mesh]'
QUERY_KEYWORDS = """
("chronic disease"[Title/Abstract] 
 OR "chronic diseases"[Title/Abstract]
 OR "type 2 diabetes"[Title/Abstract]
 OR diabetes[Title/Abstract]
 OR hypertension[Title/Abstract]
 OR "coronary artery disease"[Title/Abstract]
 OR "heart failure"[Title/Abstract]
 OR stroke[Title/Abstract]
 OR "chronic kidney disease"[Title/Abstract]
 OR copd[Title/Abstract]
 OR asthma[Title/Abstract]
 OR cancer[Title/Abstract]
)
"""
QUERY = f"({QUERY_MESH}) OR ({QUERY_KEYWORDS})"

# 近 5 年：按当前年份自动算
from datetime import datetime
current_year = datetime.now().year
start_year = current_year - 5 + 1   # 如现在是 2025，则 start_year=2021

START_DATE = date(start_year, 1, 1)
END_DATE   = date(current_year, 12, 31)

# 每个时间片中允许的最大文献数：< 9999，避免 ESearch 截断
MAX_PER_INTERVAL = 9000

# EFetch 每次拉多少篇
BATCH_SIZE = 200

# 请求间隔（秒），别太快，避免被限流
SLEEP_SEC = 0.34

# （选填）本次运行最多抓多少篇（防止一次拉太多）
# None 表示不限制
MAX_RECORDS_THIS_RUN = None


# ========== 工具函数 ==========

def esearch_count(mindate_str: str, maxdate_str: str) -> int:
    """只查询某个日期区间的总数 count"""
    handle = Entrez.esearch(
        db="pubmed",
        term=QUERY,
        datetype="pdat",
        mindate=mindate_str,
        maxdate=maxdate_str,
        retmax=0,   # 只要 count
    )
    record = Entrez.read(handle)
    handle.close()
    count = int(record["Count"])
    return count


def esearch_ids(mindate_str: str, maxdate_str: str, retmax: int) -> list:
    """在给定日期区间内获取最多 retmax 篇文献的 PMIDs（保证 retmax <= 9999）"""
    handle = Entrez.esearch(
        db="pubmed",
        term=QUERY,
        datetype="pdat",
        mindate=mindate_str,
        maxdate=maxdate_str,
        retmax=retmax,
    )
    record = Entrez.read(handle)
    handle.close()
    ids = record.get("IdList", [])
    return ids


def date_to_str(d: date) -> str:
    """date 转成 PubMed 需要的 YYYY/MM/DD 格式"""
    return d.strftime("%Y/%m/%d")


def split_interval(start: date, end: date) -> list:
    """
    递归划分日期区间：
    - 如果当前区间内文献数量 <= MAX_PER_INTERVAL，则返回这个区间
    - 否则，将区间对半拆分（start~mid, mid+1~end），递归继续
    """
    mindate_str = date_to_str(start)
    maxdate_str = date_to_str(end)
    count = esearch_count(mindate_str, maxdate_str)
    print(f"Interval {mindate_str} ~ {maxdate_str} : count = {count}")

    # 如果总数为 0，直接跳过
    if count == 0:
        return []

    # 如果数量在允许范围内，直接用这个区间
    if count <= MAX_PER_INTERVAL or start == end:
        return [(start, end, count)]

    # 否则拆成两半
    delta = (end - start).days
    if delta <= 0:
        # 按理不会到这里，保险
        return [(start, end, count)]

    mid = start + timedelta(days=delta // 2)
    left_intervals = split_interval(start, mid)
    right_intervals = split_interval(mid + timedelta(days=1), end)
    return left_intervals + right_intervals


def efetch_by_ids(id_list: list, max_retries: int = 3) -> list:
    """根据一批 PMIDs 获取详细信息，失败时自动重试几次"""
    if not id_list:
        return []

    for attempt in range(1, max_retries + 1):
        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(id_list),
                rettype="medline",
                retmode="xml",
            )
            records = Entrez.read(handle)
            handle.close()

            results = []
            for article in records.get("PubmedArticle", []):
                pmid = str(article["MedlineCitation"]["PMID"])
                article_data = article["MedlineCitation"]["Article"]

                # 标题
                title = str(article_data.get("ArticleTitle", ""))

                # 摘要
                abstract_list = article_data.get("Abstract", {}).get("AbstractText", [])
                abstract = " ".join(str(x) for x in abstract_list)

                # 年份
                year = None
                pub_date = article_data.get("ArticleDate", [])
                if pub_date:
                    year = pub_date[0].get("Year")
                if not year:
                    journal = article_data.get("Journal", {})
                    journal_issue = journal.get("JournalIssue", {})
                    pub_date2 = journal_issue.get("PubDate", {})
                    year = pub_date2.get("Year")

                results.append(
                    {
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "year": year,
                        "disease_group": "chronic_mixed",
                    }
                )

            return results

        except (IncompleteRead, HTTPError, URLError, socket.timeout) as e:
            print(f"  [WARN] EFetch 失败（第 {attempt}/{max_retries} 次）：{e}")
            time.sleep(0.5 * attempt)
        except Exception as e:
            print(f"  [WARN] EFetch 发生未知错误（第 {attempt}/{max_retries} 次）：{e}")
            time.sleep(0.5 * attempt)

    print("  [ERROR] EFetch 在多次重试后仍失败，跳过这一批 PMIDs。")
    return []




def load_existing_pmids() -> set:
    """从已有 CSV 中读取已经抓过的 pmid，避免重复写入"""
    if not OUT_FILE.exists():
        return set()

    pmids = set()
    with OUT_FILE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmid = row.get("pmid")
            if pmid:
                pmids.add(pmid)
    print(f"已有 CSV 中已有 {len(pmids)} 条 pmid（将用来去重）")
    return pmids


def append_to_csv(records: list, existing_pmids: set) -> int:
    """把 records 追加写入 CSV，按 pmid 去重，返回新写入条数"""
    if not records:
        return 0

    file_exists = OUT_FILE.exists()
    mode = "a" if file_exists else "w"
    written = 0

    with OUT_FILE.open(mode, encoding="utf-8", newline="") as f:
        fieldnames = ["pmid", "title", "abstract", "year", "disease_group"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for r in records:
            pmid = r["pmid"]
            if pmid in existing_pmids:
                continue
            writer.writerow(r)
            existing_pmids.add(pmid)
            written += 1

    return written

import json
from datetime import datetime

def save_intervals_cache(intervals):
    """
    把划分好的日期区间缓存到本地 JSON 文件，避免每次都重新递归 split_interval。
    intervals: List[(start_date, end_date, count)]
    """
    data = {
        "query": QUERY,
        "mindate": date_to_str(START_DATE),
        "maxdate": date_to_str(END_DATE),
        "max_per_interval": MAX_PER_INTERVAL,
        "intervals": [
            {
                "start": d_start.isoformat(),  # "2021-01-01"
                "end": d_end.isoformat(),      # "2021-01-03"
                "count": count,
            }
            for d_start, d_end, count in intervals
        ],
    }
    with INTERVALS_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已将区间信息缓存到: {INTERVALS_FILE}")


def load_intervals_cache():
    """
    尝试从 JSON 文件加载区间。
    如果文件不存在，或 query / 日期范围 / MAX_PER_INTERVAL 不匹配，就返回 None。
    """
    if not INTERVALS_FILE.exists():
        return None

    try:
        with INTERVALS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取区间缓存文件失败，将重新划分区间: {e}")
        return None

    # 检查参数是否一致
    if (
        data.get("query") != QUERY
        or data.get("mindate") != date_to_str(START_DATE)
        or data.get("maxdate") != date_to_str(END_DATE)
        or data.get("max_per_interval") != MAX_PER_INTERVAL
    ):
        print("缓存中的 query / 日期范围 / MAX_PER_INTERVAL 与当前设置不一致，将重新划分区间。")
        return None

    intervals_raw = data.get("intervals", [])
    intervals = []
    for item in intervals_raw:
        try:
            d_start = datetime.fromisoformat(item["start"]).date()
            d_end = datetime.fromisoformat(item["end"]).date()
            count = int(item["count"])
            intervals.append((d_start, d_end, count))
        except Exception as e:
            print(f"解析缓存区间时出错: {e}，将重新划分区间。")
            return None

    print(f"已从缓存读取 {len(intervals)} 个日期区间。")
    return intervals


def get_intervals():
    """
    优先从缓存读取区间；如果没有缓存或不匹配，则重新递归划分并写缓存。
    """
    intervals = load_intervals_cache()
    if intervals is not None:
        return intervals

    print("=== Step 1: 递归划分日期区间，确保每个区间 count <= MAX_PER_INTERVAL ===")
    intervals = split_interval(START_DATE, END_DATE)
    save_intervals_cache(intervals)
    return intervals

# ========== 主流程 ==========

def main():
    # ✅ 第一步：优先从缓存读取区间，没有再递归划分
    intervals = get_intervals()
    total_estimated = sum(c for _, _, c in intervals)
    print(f"\n总共有 {len(intervals)} 个日期区间，估计总文献数 ≈ {total_estimated}")

    # 载入已有 pmid 集合，用于去重（支持多次运行）
    existing_pmids = load_existing_pmids()

    total_written = 0           # 本次运行中新写入 CSV 的条数
    processed_records = 0       # 本次运行中“处理过的 ID 数”（不管是否已存在）

    # 如果只想本次运行抓一部分，可以用这个限制
    max_this_run = MAX_RECORDS_THIS_RUN if MAX_RECORDS_THIS_RUN is not None else float("inf")

    print("\n=== Step 2: 遍历每个时间区间，拉 PMIDs + EFetch 详情 ===")
    for idx, (start, end, count) in enumerate(intervals, start=1):
        if processed_records >= max_this_run:
            print("本次运行达到 MAX_RECORDS_THIS_RUN 限制，提前结束。")
            break

        mindate_str = date_to_str(start)
        maxdate_str = date_to_str(end)
        print(f"\n[Interval {idx}/{len(intervals)}] {mindate_str} ~ {maxdate_str}, count ≈ {count}")

        # 为安全起见，再算一遍，防止中途 PubMed 数据更新
        real_count = esearch_count(mindate_str, maxdate_str)
        print(f"  实际当前 count = {real_count}")
        if real_count == 0:
            continue

        # 获取该时间区间内的全部 PMIDs（real_count 肯定 <= MAX_PER_INTERVAL）
        ids = esearch_ids(mindate_str, maxdate_str, retmax=real_count)
        print(f"  拿到 {len(ids)} 个 PMIDs。")

        # ✅ 新增：先用 existing_pmids 过滤，看看这个区间有没有“新”的 pmid
        missing_ids = [pmid for pmid in ids if pmid not in existing_pmids]
        if not missing_ids:
            print("  该时间区间内的所有 PMIDs 已经在 CSV 中，跳过 EFetch。")
            continue

        print(f"  其中有 {len(missing_ids)} 个 PMIDs 尚未抓取，开始分批 EFetch ...")

        # 分批 EFetch 这些“缺失”的 pmid
        for start_idx in range(0, len(missing_ids), BATCH_SIZE):
            if processed_records >= max_this_run:
                print("  已达到本次运行处理上限，停止本区间后续抓取。")
                break

            batch_ids = missing_ids[start_idx:start_idx + BATCH_SIZE]
            print(f"    - EFetch {start_idx + 1} ~ {start_idx + len(batch_ids)} ...")
            records = efetch_by_ids(batch_ids)  # 记得你这里用的是带重试的版本
            written = append_to_csv(records, existing_pmids)
            print(f"      写入 CSV 新增 {written} 条（去重后）")

            processed_records += len(batch_ids)   # 按处理的 ID 数统计
            total_written += written

            time.sleep(SLEEP_SEC)

    print("\n=== 抓取结束 ===")
    print(f"本次运行 EFetch 处理了约 {processed_records} 条（按 ID 数计），")
    print(f"其中新写入 CSV 的条数为 {total_written}。")
    print(f"当前 CSV 中共有 {len(existing_pmids)} 条记录。")


if __name__ == "__main__":
    main()
