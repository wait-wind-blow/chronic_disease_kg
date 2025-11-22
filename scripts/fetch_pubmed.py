from Bio import Entrez
import pandas as pd
from pathlib import Path
import time

# 1. 配置你的邮箱（NCBI 要求）
Entrez.email = "2740142339@qq.com"  # TODO: 改成你自己的邮箱

# 2. 输出目录
BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "data" / "raw" / "pubmed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 3. 定义四大病种的检索式
QUERIES = {
    "diabetes": '''("Diabetes Mellitus, Type 2"[MeSH Terms] OR "type 2 diabetes"[Title/Abstract])
AND (diet[Title/Abstract] OR nutrition[Title/Abstract] OR "medical nutrition therapy"[Title/Abstract] OR lifestyle[Title/Abstract])
AND (humans[MeSH Terms])
AND ("2015"[PDAT] : "3000"[PDAT])''',

    "cardiovascular": '''("Cardiovascular Diseases"[MeSH Terms] OR cardiovascular[Title/Abstract] OR "coronary heart disease"[Title/Abstract] OR "myocardial infarction"[Title/Abstract] OR stroke[Title/Abstract] OR hypertension[Title/Abstract])
AND (diet[Title/Abstract] OR nutrition[Title/Abstract] OR "physical activity"[Title/Abstract] OR exercise[Title/Abstract] OR obesity[Title/Abstract])
AND (humans[MeSH Terms])
AND ("2015"[PDAT] : "3000"[PDAT])''',

    "respiratory": '''("Pulmonary Disease, Chronic Obstructive"[MeSH Terms] OR COPD[Title/Abstract] OR Asthma[Title/Abstract])
AND (smoking[Title/Abstract] OR "air pollution"[Title/Abstract] OR diet[Title/Abstract] OR obesity[Title/Abstract])
AND (humans[MeSH Terms])
AND ("2015"[PDAT] : "3000"[PDAT])''',

    "cancer": '''(("Colorectal Neoplasms"[MeSH Terms] OR "colorectal cancer"[Title/Abstract])
 OR ("Breast Neoplasms"[MeSH Terms] OR "breast cancer"[Title/Abstract]))
AND (diet[Title/Abstract] OR "red meat"[Title/Abstract] OR "processed meat"[Title/Abstract] OR "whole grain"[Title/Abstract] OR "physical activity"[Title/Abstract] OR obesity[Title/Abstract])
AND (humans[MeSH Terms])
AND ("2015"[PDAT] : "3000"[PDAT])'''
}

def fetch_one_group(group_name: str, query: str, max_count: int = 200):
    """下载某一类疾病的 PubMed 文献（标题+摘要）"""
    print(f"=== Fetching {group_name} ===")

    # 1) esearch 拿 PMIDs
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_count,
        sort="relevance"
    )
    record = Entrez.read(handle)
    handle.close()
    id_list = record["IdList"]
    print(f"Found {len(id_list)} articles for {group_name}")

    # 2) efetch 拉详细信息
    time.sleep(1)
    handle = Entrez.efetch(
        db="pubmed",
        id=",".join(id_list),
        rettype="medline",
        retmode="xml"
    )
    records = Entrez.read(handle)
    handle.close()

    rows = []
    for article in records["PubmedArticle"]:
        medline = article["MedlineCitation"]
        pmid = str(medline["PMID"])
        article_data = medline["Article"]

        title = str(article_data.get("ArticleTitle", ""))
        abstract = ""
        if "Abstract" in article_data:
            # 部分文献有多个 AbstractText 段
            abstract_texts = article_data["Abstract"].get("AbstractText", [])
            abstract = " ".join(str(t) for t in abstract_texts)

        year = None
        if article_data.get("Journal") and article_data["Journal"].get("JournalIssue"):
            year = article_data["Journal"]["JournalIssue"]["PubDate"].get("Year")

        rows.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "year": year,
            "disease_group": group_name
        })

    df = pd.DataFrame(rows)
    out_file = OUT_DIR / f"pubmed_{group_name}.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df)} records to {out_file}")

if __name__ == "__main__":
    for group, q in QUERIES.items():
        fetch_one_group(group, q, max_count=200)  # 先每类抓 200 篇试试
        time.sleep(2)  # 简单限速，别把 NCBI 打挂了
