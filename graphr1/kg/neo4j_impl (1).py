# neo4j_impl.py
import os
import asyncio
from typing import Optional # 新增
from neo4j import AsyncGraphDatabase
from ..base import BaseGraphStorage

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
            #props.pop("entity_type", None)
            props["name"] = node_name 

            query = (
                f"MERGE (n {cypher_labels} {{name: $name}}) "
                "SET n += $props"
            )
            await session.run(query, name=node_name, props=props)

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
                "MATCH (a {name: $src_name}) "
                "MATCH (b {name: $tgt_name}) "
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
            return dict(record[0]) if record else None

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

    # --- 关键修复：重写 edge_degree ---
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
    # -------------------------------

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

    # 新增
    # 检查 PMID 是否存在，并更新指南列表。Neo4j 的 Node 对象只能,用来“读属性”,不能作为“写操作参数”,
    # 所以这里返回 Optional[str] 仅表示是否存在该节点
    # ---------------------原有代码---------------------
    # query = "MATCH (n:Paper {pmid: $pmid}) RETURN n"
    # result = await session.run(query, pmid=pmid)
    # record = await result.single()
    # return record["n"] if record else None
    # --------------------------------------------------
    async def get_paper_by_pmid(self, pmid: str)  -> Optional[str]:
        async with self.driver.session(database=self.database) as session:
            query = """
                    MATCH (p:Paper {pmid: $pmid})
                    RETURN p.name AS name
                    """
            result = await session.run(query, pmid=pmid)
            record = await result.single()
            return record["name"] if record else None # 返回节点名称，表示存在

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