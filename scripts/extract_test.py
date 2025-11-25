import os
import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
# ⚠️ 请替换为您在 API 易/Waitwindbl 获取的真实 Base URL 和 新 Key
API_KEY = "sk-bfpLdknKlC0pmEK6F260Ce3b02A34963802887E37e49565a"
BASE_URL = "https://api.apiyi.com/v1"  # 必须确认这个地址！
# 模型名称 (Claude 3.5 Sonnet 在该平台的映射名，通常如下，如果报错请查阅平台文档)
MODEL_NAME = "claude-3-5-sonnet-20240620"

# 输入和输出文件
INPUT_FILE = r"E:\project\chronic_disease_kg\data\gold_standard\gold_standard_to_annotate.xlsx"   # 你的金标准表格文件名
OUTPUT_FILE = "medical_data_result.xlsx"
TEXT_COLUMN = "text"  # Excel 中存放原文的列名

# 初始化客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)



# ================= Prompt 定义 =================
# (见上方，此处在代码中组装)
def get_ner_prompt(text):
    return f"""
    你是一位专业的医学文本结构化专家。
    任务：从输入文本中提取医学实体，并进行归一化和分类。

    ### 实体类型定义:
    1. "Disease": 疾病、症状
    2. "Chemical": 药物、化合物
    3. "Gene": 基因、蛋白质
    4. "Factor": 风险因素、生活习惯

    ### 输出要求:
    - 返回一个纯 JSON 列表。
    - 格式: [{{"name": "原文名", "zh_name": "中文名", "type": "类型"}}]
    - 不要使用 Markdown 代码块，直接输出 JSON 字符串。

    [Input Text]
    {text}
    """


def get_re_prompt(text, entities_json_str):
    return f"""
    你是一位医学知识图谱构建专家。
    任务：基于给定的“文本”和“已知实体”，抽取关系三元组。

    ### 输入:
    文本: "{text}"
    已知实体: {entities_json_str}

    ### 关系类型:
    Treat, Prevent, Cause, Complicate, Positive_associated, Negative_associated, Associated

    ### 规则:
    1. 仅构建已知实体间的关系。
    2. 输出纯 JSON 列表。
    3. 格式: [{{"head": "实体A", "relation": "关系", "tail": "实体B", "confidence": 0.95}}]
    - 不要使用 Markdown 代码块，直接输出 JSON 字符串。
    """


# ================= 核心函数 =================

def call_llm(prompt):
    """通用 LLM 调用函数"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # 设为0以保证最强确定性
            max_tokens=4000
        )
        content = response.choices[0].message.content.strip()
        # 清洗可能存在的 markdown 标记 (```json ... ```)
        content = content.replace("```json", "").replace("```", "").strip()
        return content
    except Exception as e:
        print(f"API 调用错误: {e}")
        return "[]"


def process_row(text):
    """处理单行数据的两步逻辑"""

    # --- Step 1: 实体抽取 (NER) ---
    ner_prompt = get_ner_prompt(text)
    ner_result_str = call_llm(ner_prompt)

    try:
        # 验证 JSON 是否合法
        entities = json.loads(ner_result_str)
        # 再次序列化以确保 Step 2 输入格式干净
        clean_entities_str = json.dumps(entities, ensure_ascii=False)
    except:
        print(f"NER JSON 解析失败，跳过 RE 步骤。返回原串: {ner_result_str[:50]}...")
        return ner_result_str, "[]"

    # 如果没有提取到实体，直接返回
    if not entities:
        return clean_entities_str, "[]"

    # --- Step 2: 关系抽取 (RE) ---
    re_prompt = get_re_prompt(text, clean_entities_str)
    re_result_str = call_llm(re_prompt)

    return clean_entities_str, re_result_str


# ================= 主程序 =================

def main():
    # 1. 读取 Excel
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到文件 {INPUT_FILE}")
        return

    df = pd.read_excel(INPUT_FILE)

    if TEXT_COLUMN not in df.columns:
        print(f"错误: 表格中找不到列名 '{TEXT_COLUMN}'，请修改代码配置。")
        return

    print(f"开始处理，共 {len(df)} 条数据...")

    # 准备新列
    extracted_entities = []
    extracted_triples = []

    # 2. 遍历处理 (使用 tqdm 显示进度条)
    for index, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row[TEXT_COLUMN])
        if len(text) < 5:  # 跳过过短文本
            extracted_entities.append("[]")
            extracted_triples.append("[]")
            continue

        entities, triples = process_row(text)
        extracted_entities.append(entities)
        extracted_triples.append(triples)

    # 3. 保存结果
    df['AI_Entities'] = extracted_entities
    df['AI_Triples'] = extracted_triples

    df.to_excel(OUTPUT_FILE, index=False)
    print(f"\n处理完成！结果已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()