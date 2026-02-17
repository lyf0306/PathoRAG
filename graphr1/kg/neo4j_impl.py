# neo4j_impl.py
import os
import asyncio
from typing import Optional
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import Neo4jError, DriverError
from ..base import BaseGraphStorage

# --- 新增: Tenacity 智能重试库 ---
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    retry_if_exception,
    before_sleep_log
)
import logging

# 定义日志记录器
logger = logging.getLogger(__name__)

# --- 新增: 定义什么样的错误需要重试 ---
def is_retryable_error(exception):
    """
    判断是否为临时的、可重试的 Neo4j 错误。
    主要关注: 约束冲突(ConstraintValidationFailed) 和 死锁(DeadlockDetected)
    """
    if not isinstance(exception, (Neo4jError, DriverError)):
        return False
    
    code = getattr(exception, "code", "") or ""
    msg = getattr(exception, "message", "") or ""
    
    # 定义需要重试的错误关键词/错误码
    retryable_codes = [
        "ConstraintValidationFailed", # 唯一性约束冲突
        "IndexEntryConflict",         # 索引冲突
        "DeadlockDetected",           # 死锁
        "LockClientStopped",          # 锁等待超时
        "RateLimit",                  # 频率限制
        "TransientError",             # 临时错误
        "ServiceUnavailable",         # 服务不可用
        "ConnectionReadTimeout"       # 连接超时
    ]
    
    # 如果错误码匹配，或者错误信息包含 connection（连接断开），则进行重试
    return any(c in code for c in retryable_codes) or "connection" in msg.lower()

class Neo4JStorage(BaseGraphStorage):
    def __init__(self, namespace: str, global_config: dict, embedding_func):
        super().__init__(namespace, global_config, embedding_func)
        self.uri = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
        self.username = os.environ.get("NEO4J_USERNAME", "neo4j")
        self.password = os.environ.get("NEO4J_PASSWORD", "neo4j")
        self.database = os.environ.get("NEO4J_DATABASE", "neo4j")
        
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri, auth=(self.username, self.password)
            )
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Neo4j. "
                f"Please check URI, username, and password in environment variables. Error: {e}"
            )

    async def close(self):
        await self.driver.close()

    # --- 修改 upsert_node: 添加 @retry 装饰器，内部逻辑保持不变 ---
    @retry(
        retry=retry_if_exception(is_retryable_error), # 只有特定的错误才重试
        wait=wait_exponential(multiplier=0.5, min=0.2, max=10) + wait_random(0, 0.5), # 指数退避+随机抖动
        stop=stop_after_attempt(10), # 最多重试10次
        reraise=True # 失败后抛出异常
    )
    async def upsert_node(self, node_name: str, node_data: dict):
        async with self.driver.session(database=self.database) as session:
            labels = []
            if "role" in node_data:
                role_label = "".join(e for e in node_data["role"] if e.isalnum())
                if role_label:
                    labels.append(role_label.capitalize())
            
            if "entity_type" in node_data:
                type_label = "".join(e for e in node_data["entity_type"] if e.isalnum())
                if type_label:
                    labels.append(type_label.capitalize())

            if not labels:
                labels.append("Node")

            cypher_labels = ":" + ":".join(labels)
            
            props = {**node_data}
            props.pop("role", None)
            # props.pop("entity_type", None) <--- 保持这一行注释状态，保留 entity_type 属性
            props["name"] = node_name 

            query = (
                f"MERGE (n {{name: $name}}) "      # 仅按名称匹配/创建
                f"SET n {cypher_labels} "          # 补全 Label (如 :Paper:Entity)
                "SET n += $props"
            )
            await session.run(query, name=node_name, props=props)

    # --- 修改 upsert_edge: 添加 @retry 装饰器 ---
    @retry(
        retry=retry_if_exception(is_retryable_error),
        wait=wait_exponential(multiplier=0.5, min=0.2, max=10) + wait_random(0, 0.5),
        stop=stop_after_attempt(10),
        reraise=True
    )
    async def upsert_edge(
        self, src_node_name: str, tgt_node_name: str, edge_data: dict
    ):
        async with self.driver.session(database=self.database) as session:
            # 默认类型
            rel_type = "RELATES_TO"
            
            # 如果 edge_data 中包含 description，尝试将其作为关系类型
            if "description" in edge_data:
                candidate_type = edge_data["description"].upper()
                # 清洗字符，确保只包含字母、数字和下划线，符合 Neo4j 命名规范
                candidate_type = "".join(e for e in candidate_type if e.isalnum() or e == '_')
                
                # 如果清洗后不为空，则使用该类型
                if candidate_type:
                    rel_type = candidate_type

            props = {**edge_data} # 复制属性
            
            # 构建 Cypher 查询
            query = (
                "MERGE (a {name: $src_name}) "
                "MERGE (b {name: $tgt_name}) "
                f"MERGE (a)-[r:{rel_type}]->(b) "
                "SET r += $props"
            )
            
            await session.run(
                query, 
                src_name=src_node_name, 
                tgt_name=tgt_node_name, 
                props=props
            )

    async def get_node(self, node_name: str):
        async with self.driver.session(database=self.database) as session:
            query = "MATCH (n {name: $name}) RETURN n"
            result = await session.run(query, name=node_name)
            record = await result.single()
            # 保持简单的容错，防止意外
            if record:
                data = dict(record[0])
                # 如果数据库中真的因为早期版本缺失了 entity_type，给个默认值防止报错
                if "entity_type" not in data:
                    data["entity_type"] = "UNKNOWN"
                return data
            return None

    async def get_edge(self, src_node_name: str, tgt_node_name: str):
        async with self.driver.session(database=self.database) as session:
            query = "MATCH (a {name: $src_name})-[r]->(b {name: $tgt_name}) RETURN r"
            result = await session.run(
                query, src_name=src_node_name, tgt_name=tgt_node_name
            )
            record = await result.single()
            return dict(record[0]) if record else None

    async def has_node(self, node_name: str) -> bool:
        async with self.driver.session(database=self.database) as session:
            query = "MATCH (n {name: $name}) RETURN count(n) as count"
            result = await session.run(query, name=node_name)
            record = await result.single()
            return record["count"] > 0

    async def has_edge(self, src_node_name: str, tgt_node_name: str) -> bool:
        async with self.driver.session(database=self.database) as session:
            query = (
                "MATCH (a {name: $src_name})-[r]->(b {name: $tgt_name}) "
                "RETURN count(r) as count"
            )
            result = await session.run(
                query, src_name=src_node_name, tgt_name=tgt_node_name
            )
            record = await result.single()
            return record["count"] > 0

    async def node_degree(self, node_name: str) -> int:
        """获取节点的度（连接数）"""
        async with self.driver.session(database=self.database) as session:
            query = "MATCH (n {name: $name})-[r]-() RETURN count(r) as degree"
            result = await session.run(query, name=node_name)
            record = await result.single()
            return record["degree"] if record else 0

    async def edge_degree(self, src_node_name: str, tgt_node_name: str) -> int:
        """获取边的度（定义为源节点度 + 目标节点度）"""
        async with self.driver.session(database=self.database) as session:
            # 修复逻辑：分步计算 d1 和 d2，最后再相加
            query = """
                MATCH (src {name: $src_name})
                OPTIONAL MATCH (src)-[r1]-()
                WITH count(r1) as d1
                MATCH (tgt {name: $tgt_name})
                OPTIONAL MATCH (tgt)-[r2]-()
                WITH d1, count(r2) as d2
                RETURN d1 + d2 as degree
            """
            result = await session.run(
                query, src_name=src_node_name, tgt_name=tgt_node_name
            )
            record = await result.single()
            return record["degree"] if record else 0

    async def get_node_edges(self, source_node_name: str):
        """获取与指定节点相连的所有边，返回 (source, target) 元组列表"""
        if not await self.has_node(source_node_name):
            return None

        async with self.driver.session(database=self.database) as session:
            query = "MATCH (n {name: $name})-[r]-(m) RETURN m.name as target"
            result = await session.run(query, name=source_node_name)
            edges = []
            async for record in result:
                edges.append((source_node_name, record["target"]))
            return edges

    async def index_done_callback(self):
        await asyncio.sleep(0.0)
        return

    async def delete_node(self, node_name: str):
        async with self.driver.session(database=self.database) as session:
            query = "MATCH (n {name: $name}) DETACH DELETE n"
            await session.run(query, name=node_name)

    async def get_paper_by_pmid(self, pmid: str)  -> Optional[str]:
        async with self.driver.session(database=self.database) as session:
            query = """
                    MATCH (p:Paper {pmid: $pmid})
                    RETURN p.name AS name
                    """
            result = await session.run(query, pmid=pmid)
            record = await result.single()
            return record["name"] if record else None

    async def update_paper_guidelines(self, pmid: str, new_guideline: str):
        async with self.driver.session(database=self.database) as session:
            # 使用 apoc 或简单的 SET 逻辑确保指南不重复
            query = """
            MATCH (n:Paper {pmid: $pmid})
            SET n.guidelines = CASE
                WHEN $new_g IN n.guidelines THEN n.guidelines
                ELSE n.guidelines + $new_g
            END
            """
            await session.run(query, pmid=pmid, new_g=new_guideline)