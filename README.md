# Apple 10-K 财报智能问答系统

## 项目简介

基于 Apple Inc. 2020-2025 年 10-K 报告构建的智能金融问答系统，使用 RAG（检索增强生成）技术实现财报信息的智能检索与问答。

### 核心功能
- 数据预处理与索引建立
- 文本语义检索
- 基于本地开源模型的问答生成
- 简洁直观的 Web 界面

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 后端框架 | FastAPI + Uvicorn |
| 前端界面 | Streamlit |
| 向量数据库 | ChromaDB |
| 嵌入模型 | sentence-transformers/all-MiniLM-L6-v2 |
| 生成模型 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |

## 快速开始

### 前置要求
- Python 3.10+
- conda（推荐）

### 方式一：本地运行

1. **创建并激活 conda 环境**
```bash
conda create -n aapl-qa-env python=3.10 -y
conda activate aapl-qa-env
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **启动后端服务**
```bash
python main.py
```

4. **启动前端界面（新终端）**
```bash
conda activate aapl-qa-env
streamlit run app.py
```

5. **访问应用**
- 前端界面：http://localhost:8501
- API 文档：http://localhost:8000/docs

### 方式二：Docker 运行

```bash
docker-compose up
```

## 使用说明

1. 在前端界面输入你的问题
2. 调整检索相关文档数量（可选）
3. 点击"获取回答"按钮
4. 查看生成的回答和参考来源

### 示例问题
- Apple 2025 年的主要产品有哪些？
- Apple 的风险因素有哪些？
- 2025 年 Apple 的营收是多少？
- Apple 的管理层讨论与分析中提到了哪些市场风险？

## 设计思路

### 1. 数据预处理
- 读取 aapl_10k.json 中的 10-K 报告数据
- 文本清洗：去除多余空白字符
- 文本分块：将长文本切分为 500 词的块，100 词重叠
- 元数据提取：年份、章节标题等

### 2. 向量存储
- 使用 sentence-transformers 生成文本向量嵌入
- 将文档块存储到 ChromaDB 向量数据库
- 支持持久化存储

### 3. 文本检索
- 基于语义相似度检索相关文档
- 返回 top-k 最相关的文档块
- 格式化检索结果为上下文

### 4. 问答生成
- 使用本地 TinyLlama 模型基于检索到的上下文生成回答
- 提示词工程：明确的指令和上下文格式

## 项目结构

```
WilliamTest/
├── data/
│   └── aapl_10k.json          # 原始 10-K 数据
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
└── README.md                  # 项目说明文档
```

## AI-Coding 协作说明

- 使用 Trae IDE 辅助开发
- Git 提交历史记录完整开发过程
- 每一个功能模块都有独立的 Git 提交

## Git 提交历史

1. Initial commit - 初始化项目结构
2. Add requirements and config - 添加依赖和配置
3. Implement data processor - 实现数据预处理
4. Implement vector store - 实现向量数据库
5. Implement retriever - 实现文本检索
6. Implement QA generator - 实现问答生成
7. Implement FastAPI backend - 实现 FastAPI 后端
8. Implement Streamlit frontend - 实现 Streamlit 前端
9. Add main entry and README - 添加主入口和 README
10. Docker configuration - 添加 Docker 配置

## 后续扩展方向

可选的优化方向：
1. 混合检索（BM25 + 语义检索）
2. 结果重排序（Reranker）
3. 多轮对话支持
4. 财务表格结构化提取
5. 多年份对比分析

## 许可证

本项目仅供学习和研究使用。
