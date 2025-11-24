from Bio import Entrez
from pathlib import Path
from datetime import date, timedelta
import csv
import time
import json
from datetime import datetime
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError
import socket

# ========== 1. PubMed 账号配置 ==========
Entrez.email = "YOUR_EMAIL_HERE"  # TODO: 记得换成你的邮箱
# Entrez.api_key = "YOUR_API_KEY_HERE" # 强烈建议申请一个 API Key 填在这里，速度翻倍

# ========== 2. 路径配置 ==========
BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "data" / "raw" / "pubmed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "pubmed_dm_cvd_5y.csv"  # 修改文件名，体现“DM_CVD”
INTERVALS_FILE = OUT_DIR / "pubmed_dm_cvd_5y_intervals.json"

# ========== 3. 检索参数 (核心修改：聚焦 T2DM + 心血管) ==========
# 逻辑：(2型糖尿病) AND (心血管疾病 OR 高血压 OR 心衰 OR 卒中 OR 冠心病)
QUERY = """
("Diabetes Mellitus, Type 2"[Mesh] OR "type 2 diabetes"[Title/Abstract] OR "T2DM"[Title/Abstract])
AND
("Cardiovascular Diseases"[Mesh] 
 OR "heart failure"[Title/Abstract] 
 OR "stroke"[Title/Abstract] 
 OR "hypertension"[Title/Abstract] 
 OR "coronary artery disease"[Title/Abstract]
 OR "atherosclerosis"[Title/Abstract]
 OR "cardiovascular"[Title/Abstract]
)
"""

# 近 5 年
current_year = datetime.now().year
start_year = current_year - 5 + 1
START_DATE = date(start_year, 1, 1)
END_DATE = date(current_year, 12, 31)

MAX_PER_INTERVAL = 9000
BATCH_SIZE = 200
SLEEP_SEC = 0.5  # 稍微调大一点，避免并发过高被封（如果你有 API Key 可以改回 0.1）

MAX_RECORDS_THIS_RUN = None


# ========== 工具函数 ==========

def esearch_count(mindate_str: str, maxdate_str: str) -> int:
    try:
        handle = Entrez.esearch(
            db="pubmed",
            term=QUERY,
            datetype="pdat",
            mindate=mindate_str,
            maxdate=maxdate_str,
            retmax=0,
        )
        record = Entrez.read(handle)
        handle.close()
        return int(record["Count"])
    except Exception as e:
        print(f"  [WARN] ESearch Error: {e}")
        time.sleep(2)
        return 0


def esearch_ids(mindate_str: str, maxdate_str: str, retmax: int) -> list:
    try:
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
        return record.get("IdList", [])
    except Exception as e:
        print(f"  [WARN] ESearch IDs Error: {e}")
        return []


def date_to_str(d: date) -> str:
    return d.strftime("%Y/%m/%d")


def split_interval(start: date, end: date) -> list:
    mindate_str = date_to_str(start)
    maxdate_str = date_to_str(end)
    count = esearch_count(mindate_str, maxdate_str)
    print(f"Checking {mindate_str} ~ {maxdate_str} : count = {count}")

    if count == 0:
        return []
    if count <= MAX_PER_INTERVAL or start == end:
        return [(start, end, count)]

    delta = (end - start).days
    mid = start + timedelta(days=delta // 2)
    return split_interval(start, mid) + split_interval(mid + timedelta(days=1), end)


def efetch_by_ids(id_list: list, max_retries: int = 3) -> list:
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
                title = str(article_data.get("ArticleTitle", ""))
                abstract_list = article_data.get("Abstract", {}).get("AbstractText", [])
                abstract = " ".join(str(x) for x in abstract_list)

                # 简单的年份提取
                pub_date = article_data.get("ArticleDate", [])
                year = pub_date[0].get("Year") if pub_date else None
                if not year:
                    # 备选方案
                    try:
                        year = article_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {}).get("Year")
                    except:
                        year = str(current_year)  # 兜底

                # 这里 disease_group 统一标记为 T2DM_CVD
                results.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "disease_group": "T2DM_CVD"
                })
            return results
        except Exception as e:
            print(f"  [WARN] EFetch Retry {attempt}: {e}")
            time.sleep(1 * attempt)
    return []


def load_existing_pmids() -> set:
    if not OUT_FILE.exists():
        return set()
    pmids = set()
    with OUT_FILE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmid = row.get("pmid")
            if pmid: pmids.add(pmid)
    return pmids


def append_to_csv(records: list, existing_pmids: set) -> int:
    if not records: return 0
    file_exists = OUT_FILE.exists()
    written = 0
    with OUT_FILE.open("a" if file_exists else "w", encoding="utf-8", newline="") as f:
        fieldnames = ["pmid", "title", "abstract", "year", "disease_group"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists: writer.writeheader()
        for r in records:
            if r["pmid"] in existing_pmids: continue
            writer.writerow(r)
            existing_pmids.add(r["pmid"])
            written += 1
    return written


# ... (save_intervals_cache, load_intervals_cache, get_intervals 代码保持不变，此处省略以节省篇幅，直接复用你原有的即可) ...
# 如果你直接复制粘贴，请保留原文件中的这两个函数，或者把它们贴回来。
# 为了方便，我把这两个函数简化写在下面，确保脚本完整运行：

def save_intervals_cache(intervals):
    data = {"query": QUERY,
            "intervals": [{"start": date_to_str(s), "end": date_to_str(e), "count": c} for s, e, c in intervals]}
    with INTERVALS_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f)


def get_intervals():
    # 简化版：每次都重新算一下吧，反正只算一次，比较稳
    return split_interval(START_DATE, END_DATE)


def main():
    intervals = get_intervals()
    print(f"\nTotal intervals: {len(intervals)}")
    existing_pmids = load_existing_pmids()

    processed = 0
    for idx, (start, end, count) in enumerate(intervals):
        print(f"\nProcessing {idx + 1}/{len(intervals)}: {start} ~ {end}, count={count}")
        ids = esearch_ids(date_to_str(start), date_to_str(end), retmax=count)
        missing = [i for i in ids if i not in existing_pmids]

        if not missing:
            print("  All exist, skipping.")
            continue

        # 分批拉取
        for i in range(0, len(missing), BATCH_SIZE):
            batch = missing[i:i + BATCH_SIZE]
            print(f"  Fetching {i} - {i + len(batch)} of {len(missing)}...")
            recs = efetch_by_ids(batch)
            w = append_to_csv(recs, existing_pmids)
            processed += len(batch)
            time.sleep(SLEEP_SEC)

    print(f"\nDone. Total processed: {processed}")


if __name__ == "__main__":
    main()