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

# 🔴 新增：Milvus 数据库配置
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
        # 输出 0 到 1 之间的门控权重 g
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
            # 兼容宏观聚合片段与单条语义片段
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

# ================= PathoLLM：双语临床特征提取 =================
async def extract_bilingual_features(patient_case, client):
    sys_prompt = (
        "你是一个极其严谨的妇科肿瘤病理与临床特征提取专家系统。\n"
        "你的唯一任务是从用户输入的复杂病历中，精准、无损地提取出决定术后辅助治疗方案的【核心特征】，并翻译为中英双语检索词。\n\n"
        "【🚨 极其重要：特征溯源与优先级法则（防误判机制）】：\n"
        "1. **【分期与诊断的绝对优先级】**：病历中通常包含多个时间节点的诊断（如外院B超、术前诊断、术中冰冻、术后病理、术后最终诊断）。你**必须、且只能**以病历文本最后或最权威的【术后诊断】或【最终病理报告】为准提取分期和级别！绝不允许提取被推翻的早期临床分期！\n"
        "2. **【淋巴结的致命重要性】**：必须仔细扫描【术后病理】或【探查记录】中的淋巴结描述。只要提到“见癌转移”、“阳性”等字眼，即使分期没写明白，也必须在检索词中加上“淋巴结阳性（Positive lymph node）”！\n"
        "3. **【检索降噪规则】**：**绝对禁止**在检索词中包含 ER、PR、Ki-67、错配修复蛋白(MLH1/MSH2等)等常规预后/激素指标，除非它们是决定分子分型的核心高危突变（如 p53突变/野生）。\n\n"
        "你只能提取以下核心维度（若病历中未提及当作“无”处理）：\n"
        "1. 疾病类型（如子宫内膜样癌）\n"
        "2. 最终组织学分级（如G1/G2/G3）\n"
        "3. 最终确定的 FIGO分期（仔细核对术后诊断，如 IIIC1期）\n"
        "4. 肌层浸润深度（如深肌层浸润）\n"
        "5. 脉管内癌栓/LVSI状态（阳性/阴性）\n"
        "6. 宫颈/子宫外受累情况及淋巴结状态（阳性/阴性）\n"
        "7. 核心突变（p53等）\n"
        "8. 严重合并症（如肥胖、2型糖尿病等，这可能触发治疗禁忌）\n\n"
        "必须严格按照以下格式输出，使用 <keywords> 标签包裹最终结果：\n"
        "<think>\n"
        "步骤1：定位术后最终诊断，确诊分期为...\n"
        "步骤2：审查淋巴结状态，结果为...\n"
        "步骤3：审查病理报告中的肌层浸润和脉管癌栓...\n"
        "</think>\n"
        "<keywords>\n"
        "[中文检索词]: 子宫内膜样癌, Ⅱ级, 浸润子宫深肌层, 脉管内癌栓阳性, 淋巴结阳性, p53野生型, FIGO IIIC1期, 2型糖尿病\n"
        "[英文检索词]: Endometrial endometrioid adenocarcinoma, Grade 2, Deep myometrial invasion, Positive LVSI, Positive lymph node, p53 wild-type, FIGO IIIC1 stage, Type 2 diabetes mellitus\n"
        "</keywords>"
    )
    try:
        response = await client.chat.completions.create(
            model=LLM_MODEL_NAME, 
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": patient_case}],
            temperature=0.0, max_tokens=2048  
        )
        raw_content = response.choices[0].message.content.strip()
        match = re.search(r'<keywords>(.*?)</keywords>', raw_content, re.DOTALL | re.IGNORECASE)
        bilingual_keywords = match.group(1).strip() if match else re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
            
        print(f"  -> 成功提取双语特征:\n{bilingual_keywords}")
        return f"术后辅助治疗方案与临床指南 (Adjuvant treatment guidelines, recommendations and management)\n{bilingual_keywords}"
    except Exception as e:
        print(f"[Warning] 提取双语检索词失败: {e}")
        return patient_case

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
    """图谱底层的粗排过滤流"""
    if not docs_list: return []
    scores = await asyncio.gather(*[compute_rerank_score(query_str, doc, rerank_client) for doc in docs_list])
    filtered_docs = [doc for doc, score in sorted(zip(docs_list, scores), key=lambda x: x[1], reverse=True) if score > 0.05]
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
            vector_storage="MilvusVectorDBStorge", # 🔴 核心修改点：这里从 NanoVectorDBStorage 改为了 MilvusStorage
            graph_storage="Neo4JStorage", reranker_func=vector_stream_reranker
        )
    except Exception as e:
        print(f"GraphR1 初始化失败: {e}")
        exit(1)

    llm_client = AsyncOpenAI(base_url=VLLM_API_URL, api_key=VLLM_API_KEY)

    patient_case = (
        """## 现病史：
患者绝经10年， 近2年每月点滴样出血，小便后擦拭可见。每年未曾体检。2020-04-16因“绝经后阴道出血”当地医院就诊，B超检查， （未见报告）。遵医嘱行宫腔镜。4-29当地医院行宫腔镜。术后病理提示：（宫腔）宫内膜显示非典型增生伴输卵管上皮化生及靴钉样改变，灶区可疑癌变。建议手术。患者为进一步治疗，我院门诊就诊，病理会诊：NH2020-01874（宫腔）子宫内膜样癌，I级，周围内膜复杂不典型增生。5-8我院妇科常规彩色超声（经阴道）检查描述:【经阴道】 子宫位置：前位；子宫大小：长径 58mm，左右径 58mm，前后径 53mm； 子宫形态：不规则；子宫回声：不均匀； 肌层彩色血流星点状， 宫腔内中低回声区24*26*19mm   宫内IUD: 无。  宫颈长度:27mm子宫前壁突起中低回声区：27*24*22mm，右后壁向外突中低回声区：42*37*33mm，左侧壁下段肌层中高回声区：33*28*28mm，余肌层数枚低回声结节，最大直径18mm右卵巢：未暴露； 左卵巢：未暴露； 【盆腔积液】：无。诊断结论:宫腔内实质占位，符合病史。子宫多发肌瘤可能。门诊建议手术治疗，拟"子宫内膜癌"收住入院。

## 既往史：
糖尿病20年，二甲双胍0.5g qn，po，格列齐特 （达美康）1#，早餐前1#，午饭后1#。平素空腹血糖＜8mmol/l。高血压20年，口服厄贝沙坦氢氯噻嗪（依伦平）1#，qd，晨服，自测血压130-150/75-80mmHg。2012年“脑梗”口服药物治疗，现左食指、拇指麻木，活动受限。
生育史：1-0-0-1

## 家族史：
否认家族性肿瘤、遗传性病史

## 术前辅助检查：
1.       B超：妇科常规彩色超声（经阴道）检查描述:【经阴道】 子宫位置：前位；子宫大小：长径 58mm，左右径 58mm，前后径 53mm； 子宫形态：不规则；子宫回声：不均匀； 肌层彩色血流星点状， 宫腔内中低回声区24*26*19mm   宫内IUD: 无。  宫颈长度:27mm子宫前壁突起中低回声区：27*24*22mm，右后壁向外突中低回声区：42*37*33mm，左侧壁下段肌层中高回声区：33*28*28mm，余肌层数枚低回声结节，最大直径18mm右卵巢：未暴露； 左卵巢：未暴露； 【盆腔积液】：无。诊断结论:宫腔内实质占位，符合病史。子宫多发肌瘤可能。
2.      上腹部CT：1.肝脏右后叶近膈顶斑片状高密度影，介入术后改变？请结合临床病史。2.双肾小结石可能；双肾囊肿可能。
3.      盆腔MRI：子宫形态尚可，呈前倾前屈位，宫体大小约6.7cm×5.8cm×4.3cm。子宫肌壁间及浆膜下可见多发结节影，最大位于子宫后壁浆膜下，大小约4.1cm*4.2cm*3.5cm，边界清晰，病灶向宫体外突出，病灶呈T1WI等低信号，T2WI低等信号，增强后轻度均匀强化，强化程度同肌层相仿。宫腔内可见一异常信号肿物影，大小约4.4cm×3.4cm×3.2cm，呈T1W等信号T2W稍高信号，DWI呈高信号，内膜肌层交界区不清，局部可达深肌层，最深处距离浆膜面约1mm。增强后肿物可见明显强化。双侧附件区未见异常信号影及异常强化灶。膀胱充盈尚可，膀胱壁完整，未见增厚，阴道、直肠未见明显异常信号。前、后陷凹内未见明显异常信号灶。增强后亦未见异常强化灶。所扫范围盆腔内及双侧腹股沟区未见明显肿大淋巴结影。影像结论：宫腔内肿物，考虑为子宫内膜癌，累及深肌层，局部可达浆膜面； 子宫多发肌瘤。
4.      NH2020-01874（宫腔）子宫内膜样癌，I级，周围内膜复杂不典型增生。
5.      CA125抗原：125.80U/ml  人附睾蛋白4：270.6pmol/L肿瘤相关： CA199抗原：568.30U/ml

## 手术：
1.腹腔镜下全子宫切除术(子宫＜10孕周)；2、腹腔镜下双侧输卵管卵巢切除
3、腹腔镜下双侧前哨淋巴结清扫术（临床试验）

子宫前位，大小4*3*3cm，形态不规则，子宫前壁见直径3cm质硬突起，右后壁向外突直径4cm质硬结节，左侧壁下段外突直径3cm质硬结节。左输卵管外观未见异常，左卵巢大小2.5*2*1.5cm，外观未见异常。右输卵管外观未见异常，右卵巢大小2*1.5*1cm，外观未见异常。其他：肠管、肝、脾、横隔下及盆壁未见明显异常。

## 术后病理：
一、全子宫
1.子宫内膜样癌，Ⅱ级，病灶大小4×3.5×2.5cm，浸润子宫深肌层；脉管内见癌栓；癌灶未累及宫颈。周围子宫内膜呈单纯萎缩性改变。
2.子宫肌壁间多发平滑肌瘤。
3.子宫局限型腺肌病。
4.慢性宫颈炎。
二、左侧卵巢包涵囊肿伴周围炎。
    右侧卵巢包涵囊肿。
三、左侧输卵管慢性炎。
    右侧输卵管周围炎。
四、（右侧前哨（超分期））淋巴结1/4枚见癌转移。
    （左侧前哨（超分期））淋巴结1/3枚见癌转移。
免疫组化：MLH1（+），MSH2（+），MSH6（+），PMS2（+），ER（+，80%，强），PR（+，30%，强），P53（野生表型），Ki-67（+，30%），PTEN（+），AE1/AE3/CD31（脉管内见癌栓），AE1/AE3/D240（脉管内见癌栓）。
（右侧前哨（超分期））AE1/AE3（见单个肿瘤细胞转移）。
（左侧前哨（超分期））AE1/AE3（见单个肿瘤细胞转移）。
（腹腔冲洗液液基细胞学）未找到恶性细胞。

## 术后诊断：
子宫内膜样癌G2 III C1期（FIGO 2009）/T1bN1（sn）M0期"""
    )
    print(f"\n>>> [3/5] 输入原始病例:\n{patient_case}")
    print("\n>>> [3.5/5] 正在调用 PathoLLM 执行双语临床特征提取...")
    enhanced_query = await extract_bilingual_features(patient_case, llm_client)
    print("\n>>> [4/5] 正在执行图谱混合检索与 MoE 动态融合...")
    
    extended_knowledge_pool = {} 
    llm_context_list = []

    try:
        param = QueryParam(mode="hybrid", top_k=40, max_token_for_text_unit=4000)
        retrieved_results = await graph_engine.aquery(enhanced_query, param)
        
        if isinstance(retrieved_results, list):
            for item in retrieved_results:
                if isinstance(item, dict):
                    content = item.get('<knowledge>', str(item)).strip()
                    graph_score = float(item.get('<coherence>', 0.0))
                    
                    if content in ["RELATES_TO", "BELONG_TO", "EVIDENCE", ""] or len(content) < 5: 
                        continue
                        
                    if content.startswith("【权威循证溯源："):
                        if "⚠️ 临床绝对警报" in content:
                            tag_info = "🚨 禁忌症熔断警报"
                        else:
                            tag_info = "🧠 图谱高阶逻辑"
                        extended_knowledge_pool[content] = {"type": tag_info, "graph_score": graph_score}
                    else:
                        extended_knowledge_pool[content] = {"type": "🧩 纯向量语义召回", "graph_score": graph_score}

            # ==========================================
            # 🚀 核心阶段：MoE 门控网络自适应打分
            # ==========================================
            fragments_to_sort = []
            
            # 1. 指挥官只看 Anchor：瞬间计算出图谱信任权重 g
            anchor_emb_np = await embedding_func(enhanced_query)
            anchor_tensor = torch.tensor(anchor_emb_np[0], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                g_weight = moe_model(anchor_tensor).item()
            print(f"\n🧠 [MoE 门控介入] 当前患者病情复杂度测算完毕，动态图谱信任权重: g = {g_weight:.3f}")
            
            candidate_contents = list(extended_knowledge_pool.keys())
            
            if candidate_contents:
                # 2. 并发计算候选方案的纯正语义分 (调用 Reranker)
                semantic_scores = await asyncio.gather(*[
                    compute_rerank_score(enhanced_query, content, rerank_client) 
                    for content in candidate_contents
                ])
                
                for idx, content in enumerate(candidate_contents):
                    info = extended_knowledge_pool[content]
                    sources = await get_source_details(graph_engine, content)
                    best_tier = min([get_guideline_tier(src.get('guidelines', '')) for src in sources] + [4])
                    
                    semantic_score = semantic_scores[idx]
                    graph_score = info["graph_score"]
                    
                    # 3. 严格执行 MoE 打分公式
                    final_score = (g_weight * graph_score) + ((1.0 - g_weight) * semantic_score)
                    
                    # 4. 绝对安全兜底：如果是禁忌症，强制置顶推给 LLM 去规避
                    if "🚨" in info["type"]:
                        final_score += 1000.0  
                        
                    # 5. 权威指南微调加成
                    tier_bonus = (4 - best_tier) * 0.1 
                    final_score += tier_bonus
                    
                    fragments_to_sort.append({
                        "content": content, "info": info, "sources": sources,
                        "best_tier": best_tier, "semantic_score": semantic_score, 
                        "graph_score": graph_score, "final_score": final_score
                    })
            
            # 按最终融合得分降序排列
            fragments_to_sort.sort(key=lambda x: x["final_score"], reverse=True)
            
            TOP_N_FOR_LLM = 10
            selected_fragments = fragments_to_sort[:TOP_N_FOR_LLM]

            print("\n" + "="*20 + f" MoE 动态融合重排证据 (Top {len(selected_fragments)}) " + "="*20)
            bibliography, ref_counter, llm_context_list = {}, 1, []
            
            for idx, frag in enumerate(selected_fragments):
                source_indices = []
                for src in frag["sources"]:
                    src_id = src['id']
                    if src_id not in bibliography:
                        src['ref_index'] = ref_counter
                        bibliography[src_id] = src
                        ref_counter += 1
                    source_indices.append(f"[{bibliography[src_id]['ref_index']}]")
                
                tier_name = TIER_NAMES[frag['best_tier']]
                ref_tag = f"【来源文献: {', '.join(source_indices)} | 证据级别: {tier_name}】" if source_indices else "【来源文献: 未知 | 证据级别: 缺乏指南支撑】"
                
                # 清晰展示 MoE 评分细节
                print(f"[{idx+1}] [{frag['info']['type']}] 复合总分: {frag['final_score']:.3f} | 图谱分: {frag['graph_score']:.3f} | 语义分: {frag['semantic_score']:.3f}")
                print(f"内容: {frag['content'][:100]}...") 
                print("-" * 30)
                llm_context_list.append(f"{ref_tag} {frag['content']}")
        
        context_str = "\n\n".join(llm_context_list)
        print("="*50)

    except Exception as e:
        print(f"检索出错: {e}")
        return

    print("\n>>> [5/5] 正在生成治疗方案...")
    system_prompt = (
        "你是一个极其严谨的妇科肿瘤 MDT（多学科会诊）辅助决策系统。\n"
        "请严格结合【原始病历】与【参考证据】制定初步的术后辅助治疗方案。\n\n"
        "【🚨 医疗安全与临床逻辑红线（绝对遵守）】：\n"
        "1. 严禁幻觉：禁止编造不存在的药物或指南中未提及的方案。\n"
        "2. 路径排他性：必须给出一条主次分明的综合推荐路径，绝不能机械叠加。\n"
        "3. 核心纠错机制（防检索偏差）：系统前端的信息提取可能存在误差，你必须**亲自重新核对原始病历中的【术后诊断】和【病理结果】**！\n"
        "4. 遇到带有【系统强制检验指令】的方案，如与真实分期冲突，坚决打入备选讨论。\n"
        "5. 【🌟 强制循证引用规则（最核心！）】：你在正文中提出的任何指南意见、治疗方案、药物或细节，**必须**在句子或段落末尾标出对应的参考文献序号（如 [1] 或 [2,3]）。仔细查看输入【参考证据】中每段话开头的 `【来源文献: [x] ...】` 标签，对应打上 `[x]`。**绝对禁止输出没有任何文献序号支持的医学断言！**\n\n"
        "【📝 输出格式约束（必须严格遵循以下 5 个模块的 Markdown 结构）】：\n"
        "1. 请先使用 <think>...</think> 进行内部循证推理分析（大声核对患者真实分期，并排查证据适用性）。\n"
        "2. 推理结束后，输出 <answer>...</answer> 结论。在 <answer> 标签内部，按照以下大纲输出：\n\n"
        "根据患者提供的病历和知识图谱检索证据，初步的 MDT 会诊意见如下：\n\n"
        "### 1. **各大主流指南推荐意见 (Guideline Perspectives)**\n"
        "（严格按照指南派系分类梳理，并纠正检索证据中的分期偏差。如无对应证据请直接忽略该项。**注意：每条意见后必须紧跟引用的文献 [x]**）\n"
        " - **ESGO指南意见**：... [x]\n"
        " - **FIGO指南意见**：... [x]\n"
        " - **NCCN 指南意见**：... [x]\n"
        " - **国内指南意见 (中华医学会妇科肿瘤学分会指南、中国肿瘤整合诊治指南)**：... [x]\n\n"
        " - **其他指南意见**：... [x]\n"
        "### 2. **MDT 综合辅助决策主路径 (Synthesized MDT Pathway)**\n"
        "（基于真实分期，给出一个最终的推荐主路径。用一行高亮代码块表达，例如：`全身系统性治疗 (Systemic therapy) + 盆腔外照射 (EBRT) ± 阴道近距离放疗 (VBT)`）\n\n"
        "### 3. **主路径方案细化与循证依据**\n"
        "（详述化疗药物选择及周期、放疗靶区及剂量。**此处每一项具体推荐的结尾都必须包含文献引用 [x]！**）\n\n"
        "### 4. **备选/暂不推荐方案讨论 (Alternatives & Exclusions)**\n"
        "（将因分期不符被废弃的方案放于此，明确指出不采纳原因）\n\n"
        "### 5. **随访计划与注意事项**\n"
        "（随访建议和副作用提示，也需附上文献引用 [x]）\n"
    )
    user_prompt = f"【患者信息】\n{patient_case}\n\n【参考证据】\n{context_str}\n\n请制定极其详细的术后辅助治疗方案。"

    try:
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            stream=True, temperature=0.0, max_tokens=2048
        )
        print("\n>>> 模型回复:\n")
        async for chunk in response:
            if content := chunk.choices[0].delta.content: print(content, end="", flush=True)
        
        print("\n\n" + "="*20 + " 参考文献 (References) " + "="*20)
        if bibliography:
            for details in sorted(bibliography.values(), key=lambda x: x['ref_index']):
                idx, pmid_val, paper_id = details['ref_index'], details.get('pmid', 'Unknown'), details['id']
                print(f"[{idx}] PMID: {pmid_val}" if pmid_val != 'Unknown' else f"[{idx}] DocID: {paper_id[:8]}... (无标准PMID文献)")
                print(f"    Title: {details['title']}\n    Guidelines: {details['guidelines']}\n" + "-" * 10)
        else: print("（本次检索未关联到具体文献节点）")
        print("="*50)

    except Exception as e: print(f"LLM 调用失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())
