from typing import List, Dict
from .vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def _compute_relevance_score(self, doc: Dict, query: str) -> float:
        """
        计算文档的相关性分数，用于 rerank
        综合考虑：距离、是否是父块、与查询的关键词匹配等
        """
        base_score = 1.0 - (doc.get('distance', 0.5) * 0.5)  # 距离越小分数越高
        
        # 父块加分
        if doc.get('metadata', {}).get('is_parent', False):
            base_score += 0.15
        
        # 关键词匹配加分
        query_lower = query.lower()
        text_lower = doc.get('text', '').lower()
        keywords = query_lower.split()
        match_count = sum(1 for keyword in keywords if keyword in text_lower)
        if match_count > 0:
            base_score += min(match_count * 0.05, 0.2)
        
        return base_score

    def _hybrid_retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        混合检索：
        1. 先检索更多文档
        2. 然后进行 rerank
        3. 返回 top_k 个结果
        """
        # 检索更多文档用于 rerank
        retrieve_k = min(top_k * 3, 20)
        results = self.vector_store.query(query, retrieve_k)
        
        retrieved_docs = []
        for i in range(len(results['ids'][0])):
            doc = {
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            }
            retrieved_docs.append(doc)
        
        # 进行 rerank
        if retrieved_docs:
            # 计算每个文档的相关性分数
            for doc in retrieved_docs:
                doc['relevance_score'] = self._compute_relevance_score(doc, query)
            
            # 按相关性分数排序
            retrieved_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # 去重（如果有父块和子块都来自同一章节，优先保留父块）
            seen_sections = set()
            filtered_docs = []
            for doc in retrieved_docs:
                section_key = (doc['metadata'].get('year'), doc['metadata'].get('section_title'))
                if section_key not in seen_sections:
                    filtered_docs.append(doc)
                    seen_sections.add(section_key)
                if len(filtered_docs) >= top_k:
                    break
            
            retrieved_docs = filtered_docs
        
        return retrieved_docs[:top_k]

    def retrieve(self, query: str, top_k: int = 5, use_hybrid: bool = True) -> List[Dict]:
        """
        检索文档
        use_hybrid: 是否使用混合检索和 rerank
        """
        if use_hybrid:
            return self._hybrid_retrieve(query, top_k)
        else:
            results = self.vector_store.query(query, top_k)
            retrieved_docs = []
            for i in range(len(results['ids'][0])):
                doc = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                retrieved_docs.append(doc)
            return retrieved_docs

    def format_context(self, results: List[Dict]) -> str:
        context_parts = []
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            chunk_type = "Parent Section" if metadata.get('is_parent', False) else "Detailed Chunk"
            context_parts.append(f"[Document {i} - {chunk_type}]")
            context_parts.append(f"Year: {metadata.get('year', 'N/A')}")
            context_parts.append(f"Section: {metadata.get('section_title', 'N/A')}")
            if 'relevance_score' in result:
                context_parts.append(f"Relevance Score: {result['relevance_score']:.3f}")
            context_parts.append(f"Content: {result['text']}")
            context_parts.append("")
        
        return "\n".join(context_parts)
