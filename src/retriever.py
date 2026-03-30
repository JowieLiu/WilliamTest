from typing import List, Dict
from .vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
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
            context_parts.append(f"[Document {i}]")
            context_parts.append(f"Year: {metadata.get('year', 'N/A')}")
            context_parts.append(f"Section: {metadata.get('section_title', 'N/A')}")
            context_parts.append(f"Content: {result['text']}")
            context_parts.append("")
        
        return "\n".join(context_parts)
