import chromadb
from sentence_transformers import SentenceTransformer
from typing import List
from .data_processor import DocumentChunk


class VectorStore:
    def __init__(self, db_path: str, model_name: str):
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_model = SentenceTransformer(model_name)
        self.collection = None

    def create_collection(self, name: str = "aapl_10k"):
        try:
            self.collection = self.client.get_collection(name=name)
        except:
            self.collection = self.client.create_collection(name=name)

    def add_documents(self, chunks: List[DocumentChunk]):
        if not self.collection:
            self.create_collection()

        if self.collection.count() > 0:
            print("Collection already contains documents, skipping addition")
            return

        ids = [chunk.id for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        embeddings = [self.get_embedding(text) for text in texts]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    def get_embedding(self, text: str):
        return self.embedding_model.encode(text).tolist()

    def query(self, query_text: str, top_k: int = 5):
        if not self.collection:
            self.create_collection()

        query_embedding = self.get_embedding(query_text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results
