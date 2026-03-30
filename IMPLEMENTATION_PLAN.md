# Apple 10-K 财报智能问答系统 - 完整实施方案

## 一、项目概述

### 1.1 项目背景
基于 Apple Inc. 2020-2025 年 10-K 报告构建智能金融问答系统，通过 RAG（检索增强生成）技术实现财报信息的智能检索与问答。

### 1.2 数据说明
- 数据源：aapl_10k.json
- 数据内容：Apple 2020-2025 年 10-K 报告，按 section 切分
- 主要章节：
  - Item 1: 公司业务描述
  - Item 1A: 风险因素
  - Item 7: 管理层讨论与分析 (MD&A)
  - Item 8: 财务报表与附注（含表格）
  - 其他：法律诉讼、高管薪酬等

### 1.3 技术要求
- 使用开源本地可加载模型（不使用 OpenAI API）
- 强调设计思路和项目完整性
- 提供 Docker 容器化
- 完整 Git 提交历史

---

## 二、系统架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户界面 (Streamlit)                      │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                   FastAPI 后端服务                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  问答接口     │  │  检索接口     │  │  管理接口     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌─────▼─────────┐  ┌───▼────────────┐
│  问答生成模块   │  │  文本检索模块   │  │  向量数据库     │
│  (本地 LLM     │  │  (语义检索     │  │  (ChromaDB)     │
└────────────────┘  └───────────────┘  └─────────────────┘
        │                    │                    │
        └────────────────────┬────────────────────┘
                             │
                    ┌────────▼─────────┐
                    │  数据预处理模块     │
                    │  (文本分块/向量化   │
                    └──────────────────┘
```

### 2.2 技术栈

| 模块 | 技术选型 | 说明 |
|------|----------|------|
| 后端框架 | FastAPI | 高性能异步 API 框架 |
| 前端界面 | Streamlit | 快速构建数据应用 |
| 向量数据库 | ChromaDB | 轻量级本地向量数据库 |
| 嵌入模型 | sentence-transformers/all-MiniLM-L6-v2 | 开源轻量级嵌入模型 |
| 生成模型 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 开源本地可运行的小模型 |

---

## 三、项目结构

```
WilliamTest/
├── data/
│   └── aapl_10k.json          # 原始数据
├── src/
│   ├── __init__.py
│   ├── data_processor.py           # 数据预处理模块
│   ├── vector_store.py           # 向量数据库模块
│   ├── retriever.py            # 文本检索模块
│   ├── qa_generator.py         # 问答生成模块
│   └── api.py                  # FastAPI 后端
├── app.py                       # Streamlit 前端
├── main.py                      # 主入口
├── requirements.txt             # Python 依赖
├── Dockerfile                 # Docker 镜像配置
├── docker-compose.yml         # Docker Compose 配置
├── .env.example               # 环境变量示例
├── README.md                  # 项目说明文档
└── .gitignore               # Git 忽略文件
```

---

## 四、详细实施步骤

### 阶段 1：项目初始化与环境配置

**步骤 1.1：创建项目结构**
```bash
mkdir -p src data
```

**步骤 1.2：创建 requirements.txt**
```txt
fastapi==0.109.0
uvicorn==0.27.0
streamlit==1.30.0
sentence-transformers==2.2.2
chromadb==0.4.22
transformers==4.36.2
torch==2.1.2
pydantic==2.5.3
python-dotenv==1.0.0
```

**步骤 1.3：创建 .env.example**
```env
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
GENERATION_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
CHROMA_DB_PATH=./chroma_db
```

**步骤 1.4：创建 .gitignore**
```
__pycache__/
*.pyc
*.pyo
*.pyd
.env
chroma_db/
*.log
```

### 阶段 2：数据预处理模块 (src/data_processor.py)

**功能：**
- 读取 aapl_10k.json
- 文本清洗与预处理
- 文本分块（chunking）
- 元数据提取

**核心代码结构：**
```python
import json
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    id: str
    text: str
    metadata: Dict
    year: int
    section_title: str

class DataProcessor:
    def __init__(self, json_path: str):
        self.json_path = json_path
    
    def load_data(self) -> List[Dict]:
        # 加载 JSON 数据
        pass
    
    def clean_text(self, text: str) -> str:
        # 文本清洗
        pass
    
    def split_into_chunks(self, text: str, chunk_size: int = 500, 
                          overlap: int = 100) -> List[str]:
        # 文本分块
        pass
    
    def process_all(self) -> List[DocumentChunk]:
        # 完整处理流程
        pass
```

### 阶段 3：向量数据库模块 (src/vector_store.py)

**功能：**
- 使用 sentence-transformers 生成向量嵌入
- 将文档块存储到 ChromaDB
- 索引管理

**核心代码结构：**
```python
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
        # 创建或获取集合
        pass
    
    def add_documents(self, chunks: List[DocumentChunk]):
        # 添加文档到向量数据库
        pass
    
    def get_embedding(self, text: str):
        # 获取文本向量
        pass
```

### 阶段 4：文本检索模块 (src/retriever.py)

**功能：**
- 语义相似度检索
- 混合检索（可选）
- 结果重排序

**核心代码结构：**
```python
from typing import List, Dict
from .vector_store import VectorStore

class Retriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        # 检索相关文档
        pass
    
    def format_context(self, results: List[Dict]) -> str:
        # 格式化检索结果为上下文
        pass
```

### 阶段 5：问答生成模块 (src/qa_generator.py)

**功能：**
- 使用本地开源模型生成回答
- 基于检索到的上下文
- 提示词工程

**核心代码结构：**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class QAGenerator:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
    
    def generate_answer(self, query: str, context: str) -> str:
        # 生成回答
        prompt = self._build_prompt(query, context)
        pass
    
    def _build_prompt(self, query: str, context: str) -> str:
        # 构建提示词
        return f"""<|system|>
You are a financial analyst. Answer the question based on the provided context.
<|user|>
Context: {context}

Question: {query}
<|assistant|>"""
```

### 阶段 6：FastAPI 后端 (src/api.py)

**功能：**
- 问答接口
- 健康检查接口
- 索引重建接口

**核心代码结构：**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="AAPL 10-K QA System")

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]

@app.post("/api/qa", response_model=QueryResponse)
async def qa_endpoint(request: QueryRequest):
    # 问答接口
    pass

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}
```

### 阶段 7：Streamlit 前端 (app.py)

**功能：**
- 用户输入问题
- 显示回答
- 显示来源
- 简洁美观界面

**核心代码结构：**
```python
import streamlit as st
import requests

st.set_page_config(page_title="AAPL 10-K 财报问答系统")

st.title("📊 Apple 10-K 财报智能问答系统")

question = st.text_input("请输入您的问题：", placeholder="例如：Apple 2025 年的营收是多少？")

if st.button("获取回答"):
    if question:
        with st.spinner("正在思考..."):
            response = requests.post(
                "http://localhost:8000/api/qa",
                json={"question": question, "top_k": 5}
            )
            if response.status_code == 200:
                result = response.json()
                st.subheader("回答：")
                st.write(result["answer"])
```

### 阶段 8：主入口 (main.py)

**功能：**
- 初始化所有组件
- 启动后端服务
- 首次运行时建立索引

**核心代码结构：**
```python
import uvicorn
from src.data_processor import DataProcessor
from src.vector_store import VectorStore
import os
from dotenv import load_dotenv

load_dotenv()

def initialize_system():
    # 初始化系统
    processor = DataProcessor("data/aapl_10k.json")
    vector_store = VectorStore(
        db_path=os.getenv("CHROMA_DB_PATH"),
        model_name=os.getenv("EMBEDDING_MODEL_NAME")
    )
    # 处理数据并建立索引
    pass

if __name__ == "__main__":
    initialize_system()
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000)
```

### 阶段 9：Docker 容器化

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
EXPOSE 8501

CMD ["sh", "-c", "python main.py & streamlit run app.py"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
      - "8501:8501"
    volumes:
      - ./chroma_db:/app/chroma_db
    environment:
      - EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
      - GENERATION_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
      - CHROMA_DB_PATH=./chroma_db
```

### 阶段 10：README.md 文档

**文档结构：**
```markdown
# Apple 10-K 财报智能问答系统

## 项目简介
基于 Apple 2020-2025 年 10-K 报告的智能金融问答系统，使用 RAG 技术实现。

## 技术栈
- FastAPI + Uvicorn
- Streamlit
- ChromaDB
- sentence-transformers
- TinyLlama

## 快速开始

### 方式一：Docker 运行
```bash
docker-compose up
```

### 方式二：本地运行
```bash
pip install -r requirements.txt
python main.py
# 另一个终端
streamlit run app.py
```

## 使用说明
访问 http://localhost:8501 即可使用。

## 设计思路
1. 数据预处理：文本清洗、分块
2. 向量存储：使用 sentence-transformers 生成嵌入
3. 检索：语义相似度检索
4. 生成：本地 TinyLlama 模型基于上下文生成回答

## AI-Coding 协作说明
- 使用 Trae IDE 辅助开发
- Git 提交历史记录开发过程
```

---

## 五、Git 提交策略

### 提交历史规划

1. **Initial commit**
   - 初始化项目结构
   - 添加 .gitignore
   - 添加 aapl_10k.json

2. **Add requirements and config**
   - requirements.txt
   - .env.example
   - Docker 配置文件

3. **Implement data processor**
   - src/data_processor.py
   - 数据加载、清洗、分块功能

4. **Implement vector store**
   - src/vector_store.py
   - 向量数据库集成

5. **Implement retriever**
   - src/retriever.py
   - 检索功能

6. **Implement QA generator**
   - src/qa_generator.py
   - 本地模型集成

7. **Implement FastAPI backend**
   - src/api.py
   - API 接口

8. **Implement Streamlit frontend**
   - app.py
   - 用户界面

9. **Add main entry and README**
   - main.py
   - README.md
   - 项目文档

10. **Docker configuration**
    - Dockerfile
    - docker-compose.yml
    - 容器化配置

---

## 六、关键指标评估

### 功能指标
- 检索准确率：相关文档是否被正确检索
- 回答相关性：回答是否基于上下文
- 响应时间：问答响应速度

### 技术指标
- 模型加载时间
- 索引建立时间
- 内存/CPU 使用率

---

## 七、项目验收清单

- [ ] 完整的项目结构
- [ ] 数据预处理模块
- [ ] 向量数据库集成
- [ ] 文本检索功能
- [ ] 本地模型问答生成
- [ ] FastAPI 后端服务
- [ ] Streamlit 前端界面
- [ ] Docker 容器化
- [ ] 完整的 Git 提交历史
- [ ] README.md 文档

---

## 八、后续扩展方向

可选的优化方向：
1. 混合检索（BM25 + 语义检索）
2. 结果重排序（Reranker）
3. 多轮对话支持
4. 财务表格结构化提取
5. 多年份对比分析
