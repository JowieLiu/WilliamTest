import os
os.environ["CHROMA_TELEMETRY"] = "0"

import uvicorn
from dotenv import load_dotenv
from src.data_processor import DataProcessor
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.qa_generator import create_qa_generator
from src.api import app, initialize_components

load_dotenv()


def initialize_system():
    print("Initializing system...")
    
    data_path = "data/aapl_10k.json"
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    model_type = os.getenv("GENERATION_MODEL_TYPE", "tinyllama")
    chroma_db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    
    print(f"Loading data from {data_path}...")
    processor = DataProcessor(data_path)
    chunks = processor.process_all()
    print(f"Processed {len(chunks)} document chunks")
    
    print(f"Initializing vector store at {chroma_db_path}...")
    vector_store = VectorStore(chroma_db_path, embedding_model_name)
    vector_store.create_collection()
    
    print("Adding documents to vector store...")
    vector_store.add_documents(chunks)
    print("Documents added successfully!")
    
    print("Initializing retriever...")
    retriever = Retriever(vector_store)
    
    print(f"Creating QA generator (type: {model_type})...")
    qa_generator = create_qa_generator(model_type)
    
    print("Initializing API components...")
    initialize_components(vector_store, retriever, qa_generator)
    
    print("System initialization complete!")
    return vector_store, retriever, qa_generator


if __name__ == "__main__":
    initialize_system()
    print("Starting FastAPI server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
