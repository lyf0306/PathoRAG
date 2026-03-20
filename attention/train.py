import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ================= 1. 全局配置区域 =================
TRAIN_DATA_FILE = "/root/result/moe_training_data_v2_subgraph.json" 
MODEL_SAVE_PATH = "/root/Model/clinical_attention_v2.pth"          

EMBEDDING_MODEL_PATH = "/root/Model/Qwen3-Embedding-4B"
BATCH_SIZE = 32
EPOCHS = 40           # 网络变深了，稍微增加几轮 Epoch 保证充分收敛
LEARNING_RATE = 2e-4  # 🌟 降低学习率，防止深层网络早期梯度震荡
MARGIN = 0.4          

# 🌟 重装升级：多头配置翻倍
NUM_HEADS = 8         # 8个独立临床意图头
HEAD_DIM = 128        # 每个头 128 维超高分辨率

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROLE_VOCAB = {"EVIDENCE": 0, "CONTEXT": 1, "CONDITION": 2, "RECOMMENDATION": 3, "CONTRAINDICATION": 4, "PAD": 5}
MAX_ENTITIES = 40 
# ===============================================

class HypergraphAttentionDataset(Dataset):
    def __init__(self, data_file, embedding_model):
        print(">>> 正在解析超图数据集...")
        with open(data_file, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
            
        self.queries = [item["anchor_text"] for item in self.raw_data]
        self.emb_dim = embedding_model.get_sentence_embedding_dimension()
        
        unique_entities = set()
        for item in self.raw_data:
            for ent in item.get("positive_subgraph", []): unique_entities.add(ent["entity"])
            for ent in item.get("negative_subgraph", []): unique_entities.add(ent["entity"])
        unique_entities = list(unique_entities)
        
        print(f">>> 预计算 {len(unique_entities)} 个实体的 Embedding (请稍候)...")
        with torch.no_grad():
            self.query_embs = embedding_model.encode(self.queries, convert_to_tensor=True).cpu()
            if len(unique_entities) > 0:
                ent_embs = embedding_model.encode(unique_entities, convert_to_tensor=True).cpu()
                self.ent_emb_map = {ent: ent_embs[i] for i, ent in enumerate(unique_entities)}
            else:
                self.ent_emb_map = {}

        print(">>> 正在组装 Role 与 IDF 张量...")
        self.pos_data, self.neg_data = [], []
        for item in self.raw_data:
            self.pos_data.append(self._pad_subgraph(item.get("positive_subgraph", [])))
            self.neg_data.append(self._pad_subgraph(item.get("negative_subgraph", [])))

    def _pad_subgraph(self, subgraph):
        embs, roles, idfs, mask = [], [], [], []
        for ent in subgraph[:MAX_ENTITIES]:
            embs.append(self.ent_emb_map[ent["entity"]])
            roles.append(ROLE_VOCAB.get(ent["role"].upper(), 1)) 
            idfs.append(ent["idf_weight"])
            mask.append(1.0)
            
        pad_len = MAX_ENTITIES - len(embs)
        if pad_len > 0:
            embs.extend([torch.zeros(self.emb_dim)] * pad_len)
            roles.extend([ROLE_VOCAB["PAD"]] * pad_len)
            idfs.extend([0.0] * pad_len)
            mask.extend([0.0] * pad_len)
            
        return (torch.stack(embs), torch.tensor(roles, dtype=torch.long), 
                torch.tensor(idfs, dtype=torch.float), torch.tensor(mask, dtype=torch.float))

    def __len__(self): return len(self.raw_data)
    def __getitem__(self, idx): return self.query_embs[idx], self.pos_data[idx], self.neg_data[idx]

# ================= 3. 重装版：深度启发式超图注意力 =================
class DeepHeuristicHypergraphAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=8, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # 角色注入与层归一化
        self.role_embeddings = nn.Embedding(6, embedding_dim)
        nn.init.normal_(self.role_embeddings.weight, std=0.02)
        self.fusion_norm = nn.LayerNorm(embedding_dim)
        
        self.W_q = nn.Linear(embedding_dim, num_heads * head_dim)
        self.W_k = nn.Linear(embedding_dim, num_heads * head_dim)
        self.dropout = nn.Dropout(0.2)
        
        # 🌟 重装升级：多层深度交互网络 (Deep FFN)
        # 漏斗式结构，逐步提取高阶逻辑特征
        self.interaction_scorer = nn.Sequential(
            nn.Linear(4 * head_dim, 2 * head_dim),
            nn.GELU(),
            nn.LayerNorm(2 * head_dim),
            nn.Dropout(0.15),
            
            nn.Linear(2 * head_dim, head_dim),
            nn.GELU(),
            nn.LayerNorm(head_dim),
            nn.Dropout(0.1),
            
            nn.Linear(head_dim, 1) # 输出最终的交互标量
        )
        
        # 可学习的知识熵门控 (IDF Gate)
        self.idf_gate_net = nn.Linear(1, 1)
        nn.init.constant_(self.idf_gate_net.weight, 1.0)
        nn.init.constant_(self.idf_gate_net.bias, 0.0)

    def forward(self, q_emb, target_data):
        batch_size = q_emb.size(0)
        
        # ================= 场景 A：降级兼容（单文本检索） =================
        if not isinstance(target_data, (tuple, list)):
            target_emb = target_data
            default_role = torch.tensor([1], dtype=torch.long, device=target_emb.device) 
            target_emb = self.fusion_norm(target_emb + self.role_embeddings(default_role).squeeze(0))
            
            q = self.dropout(F.gelu(self.W_q(q_emb))).view(-1, self.num_heads, self.head_dim)
            k = self.dropout(F.gelu(self.W_k(target_emb))).view(-1, self.num_heads, self.head_dim)
            
            diff = torch.abs(q - k)
            mult = q * k
            interaction = torch.cat([q, k, diff, mult], dim=-1)
            scores = self.interaction_scorer(interaction).squeeze(-1)
            weights = torch.sigmoid(scores.mean(dim=-1))
            
            if self.training: return weights, q
            return weights

        # ================= 场景 B：超图结构化对齐（训练级） =================
        ent_embs, roles, idfs, mask = target_data
        max_ent = ent_embs.size(1)
        
        r_embs = self.role_embeddings(roles)
        node_features = self.fusion_norm(ent_embs + r_embs)
        
        q = self.dropout(F.gelu(self.W_q(q_emb))) 
        k = self.dropout(F.gelu(self.W_k(node_features))) 
        
        q = q.view(batch_size, 1, self.num_heads, self.head_dim)
        k = k.view(batch_size, max_ent, self.num_heads, self.head_dim)
        
        q_expanded = q.expand_as(k) 
        diff = torch.abs(q_expanded - k)
        mult = q_expanded * k
        interaction = torch.cat([q_expanded, k, diff, mult], dim=-1)
        
        # 经过深度 MLP 得到非线性基础得分 -> [B, MaxEnt, Heads]
        base_scores = self.interaction_scorer(interaction).squeeze(-1)
        
        # 动态信息熵门控
        idf_log = torch.log1p(idfs).unsqueeze(-1) 
        gate_weights = torch.sigmoid(self.idf_gate_net(idf_log)) 
        
        gated_scores = base_scores * gate_weights
        
        mask_expanded = mask.unsqueeze(-1)
        hyperedge_scores = (gated_scores * mask_expanded).sum(dim=1) 
        
        final_score = hyperedge_scores.mean(dim=-1)
        weights = torch.sigmoid(final_score)
        
        if self.training:
            return weights, q.squeeze(1)
        return weights

def orthogonal_loss(q_heads):
    q_norm = F.normalize(q_heads, p=2, dim=-1)
    sim_matrix = torch.bmm(q_norm, q_norm.transpose(1, 2)) 
    identity = torch.eye(q_heads.size(1), device=q_heads.device).unsqueeze(0)
    return (sim_matrix - identity).pow(2).sum(dim=(1, 2)).mean()

def train():
    print(f">>> 当前计算设备: {DEVICE}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device=DEVICE)
    emb_dim = embedding_model.get_sentence_embedding_dimension()
    
    dataset = HypergraphAttentionDataset(TRAIN_DATA_FILE, embedding_model)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    del embedding_model 
    torch.cuda.empty_cache()

    model = DeepHeuristicHypergraphAttention(embedding_dim=emb_dim, num_heads=NUM_HEADS, head_dim=HEAD_DIM).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.MarginRankingLoss(margin=MARGIN)

    best_loss = float('inf')
    ORTHO_WEIGHT = 0.1 
    
    print(">>> 🚀 开始训练 [重装版] 深度启发式超图注意力网络...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_pos, total_neg = 0.0, 0.0, 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for q_emb, pos_data, neg_data in progress_bar:
            q_emb = q_emb.to(DEVICE)
            pos_data = [d.to(DEVICE) for d in pos_data]
            neg_data = [d.to(DEVICE) for d in neg_data]
            
            optimizer.zero_grad()
            pos_score, q_heads_pos = model(q_emb, pos_data)
            neg_score, _ = model(q_emb, neg_data)
            
            y = torch.ones_like(pos_score).to(DEVICE)
            loss_rank = criterion(pos_score, neg_score, y)
            loss_ortho = orthogonal_loss(q_heads_pos)
            
            loss = loss_rank + ORTHO_WEIGHT * loss_ortho
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_pos += pos_score.mean().item()
            total_neg += neg_score.mean().item()
            
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "PosW": f"{pos_score.mean().item():.2f}", "NegW": f"{neg_score.mean().item():.2f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Avg Pos: {total_pos/len(dataloader):.3f} | Avg Neg: {total_neg/len(dataloader):.3f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

if __name__ == "__main__":
    train()
