# graphr1/hyper_attention.py
import os
import json
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class EndToEndHypergraphNetwork(nn.Module):
    def __init__(self, embedding_dim, num_heads=8, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        self.role_embeddings = nn.Embedding(6, embedding_dim)
        self.node_norm = nn.LayerNorm(embedding_dim)
        
        # 真正的图消息传递层 (Message Passing)
        self.msg_passing = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=8, 
            batch_first=True,
            dropout=0.1
        )
        self.post_msg_norm = nn.LayerNorm(embedding_dim)
        
        self.W_q = nn.Linear(embedding_dim, num_heads * head_dim)
        self.W_k = nn.Linear(embedding_dim, num_heads * head_dim)
        self.dropout = nn.Dropout(0.2)
        
        # 深度交叉评分网络
        self.interaction_scorer = nn.Sequential(
            nn.Linear(4 * head_dim, 2 * head_dim),
            nn.GELU(),
            nn.LayerNorm(2 * head_dim),
            nn.Dropout(0.15),
            nn.Linear(2 * head_dim, head_dim),
            nn.GELU(),
            nn.LayerNorm(head_dim),
            nn.Dropout(0.1),
            nn.Linear(head_dim, 1) 
        )

    def forward(self, q_emb, target_data):
        batch_size = q_emb.size(0)
        
        if not isinstance(target_data, (tuple, list)):
            target_emb = target_data
            default_role = torch.tensor([1], dtype=torch.long, device=target_emb.device) 
            target_emb = self.node_norm(target_emb + self.role_embeddings(default_role).squeeze(0))
            
            q = self.dropout(F.gelu(self.W_q(q_emb))).view(-1, self.num_heads, self.head_dim)
            k = self.dropout(F.gelu(self.W_k(target_emb))).view(-1, self.num_heads, self.head_dim)
            
            q_expanded = q.expand_as(k)
            interaction = torch.cat([q_expanded, k, torch.abs(q_expanded - k), q_expanded * k], dim=-1)
            
            scores = self.interaction_scorer(interaction).squeeze(-1)
            weights = torch.sigmoid(scores.mean(dim=-1))
            
            if self.training: return weights, q
            return weights
        ent_embs, roles, mask = target_data
        max_ent = ent_embs.size(1)
        
        r_embs = self.role_embeddings(roles)
        H_0 = self.node_norm(ent_embs + r_embs) 
        
        key_padding_mask = (mask == 0.0) 
        msg_out, _ = self.msg_passing(H_0, H_0, H_0, key_padding_mask=key_padding_mask)
        H_1 = self.post_msg_norm(H_0 + msg_out) 
        
        q = self.dropout(F.gelu(self.W_q(q_emb))) 
        k = self.dropout(F.gelu(self.W_k(H_1))) 
        
        q = q.view(batch_size, 1, self.num_heads, self.head_dim)
        k = k.view(batch_size, max_ent, self.num_heads, self.head_dim)
        
        q_expanded = q.expand_as(k) 
        diff = torch.abs(q_expanded - k)
        mult = q_expanded * k
        interaction = torch.cat([q_expanded, k, diff, mult], dim=-1)
        
        cross_scores = self.interaction_scorer(interaction).squeeze(-1)
        mask_expanded = mask.unsqueeze(-1).float()
        hyperedge_scores = (cross_scores * mask_expanded).sum(dim=1) 
        
        final_score = hyperedge_scores.mean(dim=-1)
        weights = torch.sigmoid(final_score)
        
        if self.training: return weights, q.squeeze(1)
        return weights

# 2. 全局单例与缓存
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ATTENTION_MODEL = None
GLOBAL_ENTITY_CACHE = {}

def init_attention_system(model_path: str, vdb_path: str, embedding_dim: int):
    global ATTENTION_MODEL, GLOBAL_ENTITY_CACHE
    if ATTENTION_MODEL is not None: return 
    
    if os.path.exists(vdb_path):
        with open(vdb_path, "r", encoding="utf-8") as f:
            vdb_data = json.load(f)
        matrix_bytes = base64.b64decode(vdb_data["matrix"])
        matrix_np = np.frombuffer(matrix_bytes, dtype=np.float32).reshape(-1, vdb_data["embedding_dim"])
        tensor_matrix = torch.tensor(matrix_np)
        
        for idx, item in enumerate(vdb_data["data"]):
            entity_name = item["entity_name"].upper()
            GLOBAL_ENTITY_CACHE[entity_name] = tensor_matrix[idx]
        print(f"  ✔ 成功加载了 {len(GLOBAL_ENTITY_CACHE)} 个实体特征的离线向量！")
    else:
        print(f"  ❌ 警告: 找不到实体向量库 {vdb_path}")

    ATTENTION_MODEL = EndToEndHypergraphNetwork(
        embedding_dim=embedding_dim, 
        num_heads=8, 
        head_dim=128
    )
    
    ATTENTION_MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
    ATTENTION_MODEL.to(DEVICE)
    ATTENTION_MODEL.eval() 

def compute_dynamic_weights_sync(query_tensor, entity_tensors):
    with torch.no_grad():
        query_tensor = query_tensor.to(DEVICE)
        entity_tensors = entity_tensors.to(DEVICE)
        weights = ATTENTION_MODEL(query_tensor.unsqueeze(0), entity_tensors)
        return weights.cpu().numpy()
