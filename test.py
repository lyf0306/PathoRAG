import os
import asyncio
import numpy as np
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from openai import AsyncOpenAI
from graphr1 import GraphR1, QueryParam
from graphr1.utils import wrap_embedding_func_with_attrs
from graphr1.hyper_attention import init_attention_system

# ================= 配置区域 =================

# LLM 服务配置
VLLM_API_URL = "http://localhost:8000/v1" 
VLLM_API_KEY = "EMPTY" 
LLM_MODEL_NAME = "OriClinical" 

# Rerank 服务配置
RERANK_API_URL = "http://localhost:8001/v1"
RERANK_API_KEY = "EMPTY"
RERANK_MODEL_NAME = "QwenReranker"

# Embedding 服务配置
EMBEDDING_API_URL = "http://localhost:8002/v1"
EMBEDDING_API_KEY = "EMPTY"
EMBEDDING_MODEL_NAME = "QwenEmbedding" 
EMBEDDING_DIM = 2560

WORKING_DIR = "/root/Graph-R1/expr/DeepSeek_QwenEmbed_Graph" 
MOE_MODEL_PATH = "/root/Model/moe_router.pth"

# Neo4j 数据库配置
os.environ["NEO4J_URI"] = "neo4j://localhost:7688"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "12345678"
os.environ["NEO4J_DATABASE"] = "neo4j"

# Milvus 数据库配置
MILVUS_URI = "http://localhost:19530"
os.environ["MILVUS_URI"] = MILVUS_URI

init_attention_system(
    model_path="/root/Model/clinical_attention_v3.pth",
    vdb_path="/root/Graph-R1/expr/DeepSeek_QwenEmbed_Graph/vdb_entities.json",
    embedding_dim=2560 
)

# ================= MoE 门控路由器网络 =================
class MoERouter(nn.Module):
    def __init__(self, input_dim):
        super(MoERouter, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        g = torch.sigmoid(self.fc3(x))
        return g

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 指南权威层级判定 =================
def get_guideline_tier(guideline_str):
    if not guideline_str: return 4
    g_str = str(guideline_str).upper()
    if "ESGO" in g_str: return 1
    elif "FIGO" in g_str: return 2
    elif "NCCN" in g_str: return 3
    else: return 4

TIER_NAMES = {
    1: "ESGO指南 (首选)", 2: "FIGO推荐 (次选)", 
    3: "NCCN指南 (推荐)", 4: "其他指南参考"
}

# ================= 获取图谱文献来源 =================
async def get_source_details(graph_engine, knowledge_text):
    sources_info = []
    try:
        storage = graph_engine.chunk_entity_relation_graph
        if not hasattr(storage, 'driver'): return sources_info
            
        async with storage.driver.session() as session:
            if knowledge_text.startswith("【权威循证溯源："):
                match = re.search(r'【权威循证溯源：(.*?)】', knowledge_text)
                if match:
                    paper_name = match.group(1).strip()
                    cypher_query = """
                    MATCH (paper:Paper)
                    WHERE paper.name = $paper_name OR paper.pmid = $paper_name
                    RETURN paper.name AS src_id, paper.pmid AS pmid, paper.title AS title, paper.guidelines AS guidelines
                    """
                    result = await session.run(cypher_query, paper_name=paper_name)
                    records = await result.data()
                else: records = []
            else:
                node_id = f"<hyperedge>{knowledge_text}"
                cypher_query = """
                MATCH (target)-[r:BELONG_TO]->(paper:Paper)
                WHERE target.name = $node_id
                RETURN paper.name AS src_id, paper.pmid AS pmid, paper.title AS title, paper.guidelines AS guidelines
                """
                result = await session.run(cypher_query, node_id=node_id)
                records = await result.data()
            
            for record in records:
                src_id = record.get("src_id", "Unknown").replace('"', '')
                raw_pmid = record.get("pmid")
                final_pmid = str(raw_pmid) if (raw_pmid and len(str(raw_pmid)) < 20) else src_id.replace("paper::", "") if "paper::" in src_id else "Unknown"
                raw_gl = record.get("guidelines")
                gl_str = ", ".join(raw_gl) if isinstance(raw_gl, list) else str(raw_gl or "General Evidence")
                
                sources_info.append({
                    "id": src_id, "pmid": final_pmid, 
                    "title": record.get("title") or "No Title", "guidelines": gl_str
                })
    except Exception as e:
        print(f"[Warning] Source lookup failed: {e}")
    return sources_info

# ================= PathoLLM：双语全息患者画像提取 =================
async def extract_patient_profile(patient_case, client):
    sys_prompt = (
        "你是一个极其严谨的妇科肿瘤病历特征提取专家系统。\n"
        "你的任务是从用户输入的 TB 报告中提取结构化信息。\n\n"
        "【🚨 核心提取法则】：\n"
        "1. 绝不能抄写模板，必须根据原始病历填入真实数据。\n"
        "2. 【分期绝对采信】：病历末尾的【术后诊断】给出了明确的 FIGO 分期，你必须直接提取，绝对禁止重新推演！\n\n"
        "请严格按照以下格式输出：\n\n"
        "### 【全息患者画像】\n"
        "- **基本体征**：[填入年龄、绝经状态、体能评分等]\n"
        "- **所有合并症与既往史**：[详尽填入所有疾病，如高血压、糖尿病、冠心病/支架、肺部炎症、胃炎等]\n"
        "- **既定分期**：[直接复制术后诊断分期，含FIGO与TNM]\n"
        "- **病理与转移特征**：\n"
        "  - 组织学类型：[填入如浆液性癌、内膜样癌等]\n"
        "  - 组织学分级：[填入 G1/G2/G3/低分化/高级别等]\n"
        "  - 浸润与周围受累：[填入肌层浸润深度、是否累及宫颈间质/输卵管/卵巢等]\n"
        "  - 脉管癌栓 (LVSI)：[填入 阳性/阴性/局灶/广泛等]\n"
        "  - 淋巴结状态：[填入转移情况及比例，如 0/2未转移]\n"
        "  - 分子分型关键指标：[重点提取 MMR(完整/缺失)、p53(野生/突变)、ER/PR、Ki-67 等]\n\n"
        "【极其重要】：在画像之后，你必须使用 <keywords> 标签提取出中英双语检索词。\n"
        "🚨【检索词提取红线】：\n"
        "1. 检索词【只允许】包含核心肿瘤特征（分期、病理类型、LVSI、淋巴结、分子分型）。【绝对禁止】在检索词中加入高血压、脑梗等非肿瘤合并症！\n"
        "2. 【英文检索词必须缩写化】：为了完美匹配国际指南图谱，英文必须使用最精炼的专业缩写与核心词簇。\n"
        "格式必须为：\n"
        "<keywords>\n"
        "[中文检索词]: 子宫内膜样癌, G2, 浸润深肌层, 脉管癌栓阳性, 淋巴结转移, MMR正常, FIGO IIIC1期\n"
        "[英文检索词]: Endometrioid, G2, deep myometrial invasion, >50% invasion, LVSI, LVSI+, node-positive, pMMR, Stage IIIC1, IIIC\n"
        "</keywords>"
    )
    
    user_prompt = f"【原始病历源数据】\n{patient_case}\n\n请提取患者画像并生成中英双语的精炼 <keywords>。"

    try:
        response = await client.chat.completions.create(
            model=LLM_MODEL_NAME, 
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0, max_tokens=2048  
        )
        raw_content = response.choices[0].message.content.strip()
        content_no_think = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
        
        profile_match = re.search(r'(### 【全息患者画像】.*?)(?=<keywords>|$)', content_no_think, re.DOTALL)
        profile_text = profile_match.group(1).strip() if profile_match else content_no_think
        
        keywords_match = re.search(r'<keywords>([\s\S]*?)</keywords>', content_no_think, re.DOTALL | re.IGNORECASE)
        if keywords_match:
            bilingual_keywords = keywords_match.group(1).strip()
        else:
            bilingual_keywords = "\n".join(content_no_think.split('\n')[-2:])
            
        print(f"  -> 成功提取全息患者画像与检索词:\n{profile_text}\n\n[检索词]:\n{bilingual_keywords}")
        
        return profile_text, bilingual_keywords
    except Exception as e:
        print(f"[Warning] 提取特征失败: {e}")
        return patient_case, patient_case

# ================= 🚀 ESGO 风险前置推理 (含自纠错机制) =================
async def evaluate_esgo_risk(patient_profile, client):
    sys_prompt = "你是一个严谨的妇科肿瘤专家。你必须严格、字对字地遵循用户提供的逻辑树进行推导，绝对不可使用你的固有记忆或跳过任何步骤。"

    user_prompt = f"""请根据以下提取出的【全息患者画像】，严格按照下方的《ESGO 2025风险判定规则库》进行分级评估。

【全息患者画像】：
{patient_profile}

【核心医学术语定义（推演必读！）】：
1. 组织学分级中，G1 和 G2 属于“低级别”（Low grade）；G3 属于“高级别”（High grade）。
2. 如果 MMR完整/正常，且 P53 为野生型，且未提及 POLE 突变，则该患者属于 NSMP 型（无特定分子谱型）。

【风险判定规则库】（必须一步步对照）：
规则一：如果分子分型为 POLE 突变型
- 低危 (Low Risk)：FIGO分期为 IA期、IB期、IC期、或 II期。
- 不确定风险 (Uncertain Risk)：FIGO分期为 III期或 IVA期。

规则二：如果分子分型为 MMRd 型
- 低危 (Low Risk)：FIGO分期为 IA期或 IC期。
- 中危 (Intermediate Risk)：FIGO分期为 IB期；或 IIC期（条件：有肌层浸润，但无宫颈间质浸润且无明显LVSI）。
- 中高危 (High-Intermediate Risk)：FIGO分期为 IIA期、IIB期；或 IIC期（伴宫颈浸润或明显LVSI）。
- 高危 (High Risk)：FIGO分期为 III期或 IVA期。

规则三：如果分子分型为 NSMP 型
子规则 3A：如果是 NSMP 低级别(G1/G2) 且 ER阳性
- 低危 (Low Risk)：FIGO分期为 IA期。
- 中危 (Intermediate Risk)：FIGO分期为 IB期或 IIA期。
- 中高危 (High-Intermediate Risk)：FIGO分期为 IIB期。
- 高危 (High Risk)：FIGO分期为 III期或 IVA期。
子规则 3B：如果是 NSMP 高级别(G3) 或 ER阴性（或两者兼有）
- 不确定风险 (Uncertain Risk)：FIGO分期为 IA1期或 IC期。
- 高危 (High Risk)：FIGO分期为 IA2期、IA3期、IB期、II期、III期或 IVA期。

规则四：如果分子分型为 p53abn 型
- 不确定风险 (Uncertain Risk)：FIGO分期为 IA1期或 IC期。
- 高危 (High Risk)：FIGO分期为 IA2期、IA3期、IB期、II期、III期或 IVA期。

【强制输出格式】：
思考结束后，你必须在正文中严格输出以下结构，且【最终结论】必须被包含在 `<result>` 标签内！
输入特征确认：...
逻辑推理过程：...
最终结论：<result>高危</result> （仅限填入：低危 / 中危 / 中高危 / 高危 / 不确定风险）"""
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            current_temp = 0.0 if attempt == 0 else 0.3
            
            response = await client.chat.completions.create(
                model=LLM_MODEL_NAME, 
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                temperature=current_temp, max_tokens=2048  
            )
            raw_content = response.choices[0].message.content.strip()
            clean_eval_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
            
            risk_level_match = re.search(r'<result>(.*?)</result>', clean_eval_content)
            extracted_risk_keyword = risk_level_match.group(1).strip() if risk_level_match else ""

            if extracted_risk_keyword and extracted_risk_keyword in ["低危", "中危", "中高危", "高危", "不确定风险", "低风险", "中风险", "高风险"]:
                risk_mapping = {
                    "低危": "Low Risk", "低风险": "Low Risk",
                    "中危": "Intermediate Risk", "中等风险": "Intermediate Risk",
                    "中高危": "High-Intermediate Risk",
                    "高危": "High Risk", "高风险": "High Risk",
                    "不确定风险": "Uncertain Risk"
                }
                en_risk_keyword = risk_mapping.get(extracted_risk_keyword, "")
                
                print(f"  -> ESGO 前置推演完成 (尝试 {attempt+1}/{max_retries})，判定结果: {extracted_risk_keyword} ({en_risk_keyword})")
                
                safe_clinical_conclusion = f"综合患者的分子分型、组织学分级、浸润深度及手术分期等病理特征，依据《2025 ESGO-ESTRO-ESP 子宫内膜癌管理指南》风险分层标准评估，该患者的复发风险等级判定为：【{extracted_risk_keyword} ({en_risk_keyword})】。"
                return safe_clinical_conclusion, extracted_risk_keyword, en_risk_keyword
            else:
                print(f"  -> [Warning] 尝试 {attempt+1}/{max_retries}: 未检测到规范结果标签，重试...")
                user_prompt += "\n\n【🚨 系统强制警告】：请重新推演，并确保最后一行严格输出例如 `<result>高危</result>` 的格式。"

        except Exception as e:
            print(f"  -> [Warning] 尝试 {attempt+1}/{max_retries} API 调用异常: {e}")
            
    print("  -> ❌ ESGO 前置推演重试达到上限，启用安全兜底隔离方案。")
    safe_clinical_conclusion = "综合患者的病理特征，根据《2025 ESGO-ESTRO-ESP指南》评估，需进一步明确其复发风险等级（系统自动推演未匹配明确分级，请结合原版指南人工复核）。"
    return safe_clinical_conclusion, "", ""

# ================= Qwen-Reranker 打分逻辑 =================
async def compute_rerank_score(query, doc, client):
    instruction = "Given a clinical case, retrieve relevant clinical guidelines and evidence that help formulate a treatment plan."
    prompt = f"<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    try:
        response = await client.completions.create(model=RERANK_MODEL_NAME, prompt=prompt, max_tokens=1, temperature=0, logprobs=20)
        top_logprobs_dict = response.choices[0].logprobs.top_logprobs[0]
        true_logit, false_logit = -10.0, -10.0
        for token_str, logprob in top_logprobs_dict.items():
            clean_token = token_str.strip().lower()
            if clean_token == "yes": true_logit = max(true_logit, logprob)
            elif clean_token == "no": false_logit = max(false_logit, logprob)
        true_score, false_score = math.exp(true_logit), math.exp(false_logit)
        return 0.0 if true_score + false_score == 0 else true_score / (true_score + false_score)
    except Exception: return 0.0

print(f">>> [1/5] 正在配置 API 客户端...")
try:
    embed_client = AsyncOpenAI(base_url=EMBEDDING_API_URL, api_key=EMBEDDING_API_KEY)
    rerank_client = AsyncOpenAI(base_url=RERANK_API_URL, api_key=RERANK_API_KEY)
    
    @wrap_embedding_func_with_attrs(embedding_dim=EMBEDDING_DIM, max_token_size=8192)
    async def embedding_func(texts):
        if isinstance(texts, str): texts = [texts]
        response = await embed_client.embeddings.create(model=EMBEDDING_MODEL_NAME, input=texts)
        return np.array([data.embedding for data in response.data])
    print("API 客户端配置成功。")
except Exception as e:
    print(f"API 客户端配置失败: {e}")
    exit(1)

async def vector_stream_reranker(query_str, docs_list):
    if not docs_list: return []
    scores = await asyncio.gather(*[compute_rerank_score(query_str, doc, rerank_client) for doc in docs_list])
    # 为了保证能召回足够的临床试验数据，放宽阈值至 0.01
    filtered_docs = [doc for doc, score in sorted(zip(docs_list, scores), key=lambda x: x[1], reverse=True) if score > 0.01]
    return filtered_docs

async def main():
    print(">>> [加载模型] 正在挂载 MoE 门控路由器...")
    moe_model = MoERouter(input_dim=EMBEDDING_DIM).to(DEVICE)
    try:
        moe_model.load_state_dict(torch.load(MOE_MODEL_PATH))
        moe_model.eval()
        print("✅ MoE 路由器挂载成功！")
    except Exception as e:
        print(f"❌ MoE 路由器挂载失败，请检查路径: {e}")
        exit(1)

    print(f">>> [2/5] 正在初始化 GraphR1 检索引擎...")
    try:
        graph_engine = GraphR1(
            working_dir=WORKING_DIR, embedding_func=embedding_func,
            kv_storage="JsonKVStorage", 
            vector_storage="MilvusVectorDBStorge", 
            graph_storage="Neo4JStorage", reranker_func=vector_stream_reranker
        )
    except Exception as e:
        print(f"GraphR1 初始化失败: {e}")
        exit(1)

    llm_client = AsyncOpenAI(base_url=VLLM_API_URL, api_key=VLLM_API_KEY)

    # 测试患者病历数据
    patient_case = (
        """## 现病史：
患者绝经10年， 近2年每月点滴样出血，小便后擦拭可见。每年未曾体检。2020-04-16因“绝经后阴道出血”当地医院就诊，B超检查， （未见报告）。遵医嘱行宫腔镜。4-29当地医院行宫腔镜。术后病理提示：（宫腔）宫内膜显示非典型增生伴输卵管上皮化生及靴钉样改变，灶区可疑癌变。建议手术。患者为进一步治疗，我院门诊就诊，病理会诊：NH2020-01874（宫腔）子宫内膜样癌，I级，周围内膜复杂不典型增生。5-8我院妇科常规彩色超声（经阴道）检查描述:【经阴道】 子宫位置：前位；子宫大小：长径 58mm，左右径 58mm，前后径 53mm； 子宫形态：不规则；子宫回声：不均匀； 肌层彩色血流星点状， 宫腔内中低回声区24*26*19mm   宫内IUD: 无。  宫颈长度:27mm子宫前壁突起中低回声区：27*24*22mm，右后壁向外突中低回声区：42*37*33mm，左侧壁下段肌层中高回声区：33*28*28mm，余肌层数枚低回声结节，最大直径18mm右卵巢：未暴露； 左卵巢：未暴露； 【盆腔积液】：无。诊断结论:宫腔内实质占位，符合病史。子宫多发肌瘤可能。门诊建议手术治疗，拟"子宫内膜癌"收住入院。

## 既往史：
糖尿病20年，二甲双胍0.5g qn，po，格列齐特 （达美康）1#，早餐前1#，午饭后1#。平素空腹血糖＜8mmol/l。高血压20年，口服厄贝沙坦氢氯噻嗪（依伦平）1#，qd，晨服，自测血压130-150/75-80mmHg。2012年“脑梗”口服药物治疗，现左食指、拇指麻木，活动受限。
生育史：1-0-0-1

## 家族史：
否认家族性肿瘤、遗传性病史

## 术后病理：
一、全子宫
1.子宫内膜样癌，Ⅱ级，病灶大小4×3.5×2.5cm，浸润子宫深肌层；脉管内见癌栓；癌灶未累及宫颈。周围子宫内膜呈单纯萎缩性改变。
2.子宫肌壁间多发平滑肌瘤。
3.子宫局限型腺肌病。
4.慢性宫颈炎。
四、（右侧前哨（超分期））淋巴结1/4枚见癌转移。
    （左侧前哨（超分期））淋巴结1/3枚见癌转移。
免疫组化：MLH1（+），MSH2（+），MSH6（+），PMS2（+），ER（+，80%，强），PR（+，30%，强），P53（野生表型），Ki-67（+，30%），PTEN（+），AE1/AE3/CD31（脉管内见癌栓），AE1/AE3/D240（脉管内见癌栓）。

## 术后诊断：
子宫内膜样癌G2 III C1期（FIGO 2009）/T1bN1（sn）M0期"""
    )
    
    print(f"\n>>> [3/5] 输入原始病例:\n")
    print("\n>>> [3.5/5] 正在调用 PathoLLM 提取全息患者画像与双语特征...")
    patient_profile_md, bilingual_keywords = await extract_patient_profile(patient_case, llm_client)

    print("\n>>> [3.6/5] 正在执行 ESGO 2025 零样本风险定级 (支持自纠错机制)...")
    safe_clinical_conclusion, esgo_risk_level, en_risk_keyword = await evaluate_esgo_risk(patient_profile_md, llm_client)
    
    # 🚀 强行扩大召回网，将试验名字植入检索词，确保底层不漏掉数据
    risk_query_addition = f"\n[ESGO推断组别]: {esgo_risk_level} ({en_risk_keyword} group)" if esgo_risk_level else ""
    enhanced_query = f"NCCN ESGO 指南、术后辅助治疗方案与预后生存率、PORTEC临床试验生存数据 (Adjuvant treatment guidelines, prognosis overall survival, clinical trials, recommendations)\n{bilingual_keywords}{risk_query_addition}"
    
    print("\n>>> [4/5] 正在执行图谱混合检索与 MoE 动态融合...")
    extended_knowledge_pool = {} 

    try:
        # 将召回数量拉高到60，给Reranker更多选择
        param = QueryParam(mode="hybrid", top_k=60, max_token_for_text_unit=4000)
        retrieved_results = await graph_engine.aquery(enhanced_query, param)
        
        if isinstance(retrieved_results, list):
            for item in retrieved_results:
                if isinstance(item, dict):
                    content = item.get('<knowledge>', str(item)).strip()
                    graph_score = float(item.get('<coherence>', 0.0))
                    if content in ["RELATES_TO", "BELONG_TO", "EVIDENCE", ""] or len(content) < 5: 
                        continue
                        
                    if content.startswith("【权威循证溯源："):
                        tag_info = "🚨 禁忌症熔断警报" if "⚠️ 临床绝对警报" in content else "🧠 图谱高阶逻辑"
                        extended_knowledge_pool[content] = {"type": tag_info, "graph_score": graph_score}
                    else:
                        extended_knowledge_pool[content] = {"type": "🧩 纯向量语义召回", "graph_score": graph_score}

            candidate_contents = list(extended_knowledge_pool.keys())
            
            if candidate_contents:
                semantic_scores = await asyncio.gather(*[
                    compute_rerank_score(enhanced_query, content, rerank_client) 
                    for content in candidate_contents
                ])
                
                fragments_to_sort = []
                for idx, content in enumerate(candidate_contents):
                    info = extended_knowledge_pool[content]
                    sources = await get_source_details(graph_engine, content)
                    best_tier = min([get_guideline_tier(src.get('guidelines', '')) for src in sources] + [4])
                    semantic_score = semantic_scores[idx]
                    
                    final_score = (0.4 * info["graph_score"]) + (0.6 * semantic_score)
                    if "🚨" in info["type"]: final_score += 1000.0  
                    final_score += (4 - best_tier) * 0.1 
                    
                    fragments_to_sort.append({
                        "content": content, "sources": sources, "final_score": final_score
                    })
            
            fragments_to_sort.sort(key=lambda x: x["final_score"], reverse=True)
            selected_fragments = fragments_to_sort[:10]

            bibliography = {}
            ref_counter = 1
            llm_context_list = []
            
            # 无条件注入 ESGO 基石文献
            esgo_paper_id = "paper::40744042"
            bibliography[esgo_paper_id] = {
                "id": esgo_paper_id, "pmid": "40744042",
                "title": "ESGO-ESTRO-ESP guidelines for the management of patients with endometrial carcinoma: update 2025",
                "guidelines": "ESGO指南", "ref_index": ref_counter
            }
            llm_context_list.append(f"【来源文献: [{ref_counter}] | 证据级别: ESGO指南 (首选)】 {safe_clinical_conclusion}")
            ref_counter += 1

            for frag in selected_fragments:
                source_indices = []
                for src in frag["sources"]:
                    src_id = src['id']
                    if src_id not in bibliography:
                        src['ref_index'] = ref_counter
                        bibliography[src_id] = src
                        ref_counter += 1
                    source_indices.append(f"[{bibliography[src_id]['ref_index']}]")
                ref_tag = f"【来源文献: {', '.join(source_indices)}】" if source_indices else ""
                llm_context_list.append(f"{ref_tag} {frag['content']}")
        
        context_str = "\n\n".join(llm_context_list)
        
        guideline_keywords = ["ESGO", "NCCN", "FIGO", "中华医学会", "CSCO", "中国肿瘤", "ESMO"]
        if not any(kw in context_str.upper() for kw in guideline_keywords):
            print("\n⚠️ [安全警报]：未命中任何国内外主流指南，触发系统防幻觉兜底策略！")
            context_str = "【系统安全诊断】：本次 RAG 未能召回 ESGO/NCCN/FIGO/中华医学会/CSCO 等权威指南文本。请在最终报告开头明确提示证据缺失，仅基于现有信息分析，并强烈建议主治医师查阅原版指南。\n\n" + context_str
            
        print("="*50)

    except Exception as e:
        print(f"检索出错: {e}")
        return

    # ================= 🚀 终极“分工协同版” MDT 生成 Prompt (指南极尽详尽，数据留给下游) =================
    system_prompt = "你是一名严谨的妇科肿瘤 MDT 首席专家。你的任务是：基于知识图谱提供的证据，对【权威指南推荐】进行极其详尽的拆解和论述；同时，为下游的 EBM（循证医学）多智能体系统提出需要深度查证的具体临床数据问题。严禁凭空捏造具体的生存率或 HR 数值。"
    
    user_prompt = f"""以下是患者的【全息特征画像】：
{patient_profile_md}

以下是图谱召回的【参考证据】（已包含系统前置推演结论及权威文献）：
{context_str}

【🚨 核心分工与防幻觉准则】：
1. **指南解析必须极尽详尽（核心任务！）**：下游的 EBM 系统无法阅读长篇的指南全文本。因此，你必须把你能在【参考证据】中看到的关于 ESGO、NCCN、国内共识的具体治疗细则（如：满足什么条件推荐什么放疗、化疗用什么方案等），**长篇大论、毫无遗漏地写出来，并打上标号 [x]**。
2. **数据验证留给下游**：对于具体的临床试验（如 PORTEC-3 的具体 5 年 OS 百分比、HR 值），如果图谱证据里有，你可以简述；如果没有，**绝对禁止编造**，直接把获取这些精准数据的任务写在第四部分的“待查证问题”中！
3. **拒绝张冠李戴**：认清指南名称，没检索到纯正的 NCCN 原文，就写国内指南或 ESGO 结论，切勿伪造。

【📝 强制报告结构（请严格按照以下模块输出）】：

# 妇科肿瘤 MDT 初始会诊报告

## 一、 病情摘要与风险判定
- **病历摘要**：凝练患者年龄、绝经史、合并症、核心病理、淋巴结状态及 FIGO 分期。
- **ESGO 2025 风险分层**：直接引用 [1] 号文献的结论（如：**高危 (High Risk)** [1]）。

## 二、 核心指南与共识详尽解析
（🚨 **本段是报告的灵魂，必须长篇详写！**）
- **指南主干路径**：用代码块高亮（如 `系统治疗 ± EBRT ± VBT`）。
- **ESGO 指南详细解析**：展开论述 [1] 号文献对于该风险组的具体干预意见。
- **其他指南详尽解析**：详细罗列检索到的 NCCN 或国内共识的具体细则（如盆腔外照射的具体指征、淋巴结阳性时的具体推荐等），必须带上真实文献标号 [x]。

## 三、 初步专科治疗框架
- **肿瘤主方案建议**：给出放化疗的大体建议及影像学复查节点。
- **多学科及合并症管理**：
  （**🚨 必须使用阿拉伯数字分点列出所有的合并症！绝不能合并成一段！**）
  1、患者高血压，建议心内科随诊。
  2、患者糖尿病，建议内分泌科随诊，关注化疗期间血糖波动。
  3、患者脑梗后遗症，建议神经内科随诊，注意血栓风险。
  ……（逐条列出）

## 四、 随访大纲
列出常规的随访频率（如前两年每3个月一次）以及需要患者警惕的核心异常体征（如提示粘连梗阻或复发的症状）、检查项目等。

## 五、 🎯 待 PathoEBM 深度合成的临床问题（交接任务）
（作为主治医师，请针对本病例的特殊性，向你的下游“EBM循证医学AI助手”提出 2~3 个需要它通过检索 PubMed 等数据库去深度查证的 PICO 问题。
**提问思路示例**：
- 请检索 PORTEC-3 等大型 RCT 的最新长期随访数据，明确该分期/分子分型患者接受放化疗联合的具体 OS/DFS 获益百分比及 HR 值。
- 患者有20年糖尿病和脑梗史，请查阅真实世界数据（RWS），评估高龄合并严重心脑血管基础病的内膜癌患者，采用 TC 方案化疗的心血管/神经毒性发生率，并提供剂量调整的前沿文献。
- 检索关于 L1CAM 等新兴标志物在此类高危患者中对预后影响的最新前沿研究。）

💡 请先在 <think> 标签内梳理：1. 确认风险等级；2. 疯狂提取证据里的指南细节；3. 盘点所有合并症；4. 构思要留给下游的硬核数据问题。思考完毕后，输出这份专业报告！"""

    try:
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": user_prompt}
            ],
            stream=True, 
            temperature=0.2,       
            max_tokens=8192        # 恢复 8192，给模型充足的 Token 去长篇大论地拆解指南
        )
        print("\n>>> 模型回复:\n")
        async for chunk in response:
            if content := chunk.choices[0].delta.content: 
                print(content, end="", flush=True)
        
        print("\n\n" + "="*20 + " 参考文献 (References) " + "="*20)
        if bibliography:
            for details in sorted(bibliography.values(), key=lambda x: x['ref_index']):
                idx, pmid_val, paper_id = details['ref_index'], details.get('pmid', 'Unknown'), details['id']
                print(f"[{idx}] PMID: {pmid_val}" if pmid_val != 'Unknown' else f"[{idx}] DocID: {paper_id[:8]}... (无标准PMID文献)")
                print(f"    Title: {details['title']}\n    Guidelines: {details['guidelines']}\n" + "-" * 10)
        else: 
            print("（本次检索未关联到具体文献节点）")
        print("="*50)

    except Exception as e: 
        print(f"LLM 调用失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())
