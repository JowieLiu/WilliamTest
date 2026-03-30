# Apple 10-K 财报智能问答系统

## 项目简介

基于 Apple Inc. 2020-2025 年 10-K 报告构建的智能金融问答系统，使用 RAG（检索增强生成）技术实现财报信息的智能检索与问答。

### 核心功能
- 数据预处理与索引建立
- 父子切块混合检索
- 结果 Rerank 重排序
- 灵活的模型配置（本地小模型 / API 大模型）
- 简洁直观的 Web 界面
- 流式输出（实时显示处理进度）
- 用户反馈系统
- 完整的测试套件

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 后端框架 | FastAPI + Uvicorn |
| 前端界面 | Streamlit |
| 向量数据库 | ChromaDB |
| 嵌入模型 | sentence-transformers/all-MiniLM-L6-v2 |
| 生成模型 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 / OpenAI API |
| 测试框架 | pytest |

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

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/test_system.py -v

# 运行测试并生成覆盖率报告
pip install pytest-cov
python -m pytest tests/test_system.py --cov=src --cov-report=html
```

## 使用说明

### 基本使用
1. 在前端界面输入你的问题
2. 调整检索相关文档数量（可选）
3. 点击"获取回答"按钮
4. 查看生成的回答和参考来源

### 模型配置
- **默认模式**：使用本地 TinyLlama 小模型进行总结
- **API 模式**：在侧边栏配置 API URL、API Key 和模型名称，切换到 API 大模型

### 示例问题
- Apple 2025 年的主要产品有哪些？
- Apple 的风险因素有哪些？
- 2025 年 Apple 的营收是多少？
- Apple 的管理层讨论与分析中提到了哪些市场风险？

## 设计思路

### 1. 数据预处理 - 父子切块
- 读取 aapl_10k.json 中的 10-K 报告数据
- 文本清洗：去除多余空白字符
- **父子切块策略**：
  - **父块**：包含完整章节的摘要（前 1000 字符）
  - **子块**：详细内容切块（350 词，50 词重叠）
- 元数据提取：年份、章节标题、块类型等

### 2. 向量存储
- 使用 sentence-transformers 生成文本向量嵌入
- 将文档块存储到 ChromaDB 向量数据库
- 支持持久化存储
- HNSW 索引优化检索速度

### 3. 文本检索 - 混合检索 + Rerank
- **混合检索**：先检索更多文档（top_k * 3）用于 rerank
- **Rerank 重排序**：
  - 基于向量距离计算基础分数
  - 父块加分（+0.15）
  - 关键词匹配加分（每个匹配关键词 +0.05，最高 +0.2）
- **去重策略**：同一章节优先保留父块
- 格式化检索结果为上下文

### 4. 问答生成
- **灵活的生成器架构**：策略模式 + 工厂模式
- **多种生成器**：
  - FallbackGenerator：直接展示检索结果
  - TinyLlamaGenerator：本地小模型总结
  - OpenAIApiGenerator：API 大模型调用
- **提示词优化**：
  - 明确的系统指令
  - 要求基于上下文总结
  - 结构化的上下文格式
- **全局生成器管理**：支持运行时动态切换

### 5. 问答生成
- **灵活的生成器架构**：策略模式 + 工厂模式
- **多种生成器**：
  - FallbackGenerator：直接展示检索结果
  - TinyLlamaGenerator：本地小模型总结
  - OpenAIApiGenerator：API 大模型调用
- **提示词优化**：
  - 明确的系统指令
  - 要求基于上下文总结
  - 结构化的上下文格式
- **全局生成器管理**：支持运行时动态切换

### 6. 流式输出
- Server-Sent Events (SSE) 协议
- 逐词模拟流式输出
- 先发送参考来源，再发送答案
- 支持完成信号和错误处理

### 7. 用户反馈系统
- 简单的反馈界面（👍/👎）
- 可选的评论输入
- 内存存储反馈数据（保留最近 1000 条）
- 管理接口获取反馈统计

### 8. 测试套件
- 单元测试：覆盖所有核心模块
- 集成测试：端到端完整流程测试
- Mock 测试：API 调用使用 mock 避免网络请求

## 项目结构

```
WilliamTest/
├── data/
│   └── aapl_10k.json          # 原始 10-K 数据
├── src/
│   ├── __init__.py
│   ├── data_processor.py      # 数据预处理模块（父子切块）
│   ├── vector_store.py        # 向量数据库模块
│   ├── retriever.py           # 文本检索模块（混合检索 + Rerank）
│   ├── qa_generator.py        # 问答生成模块（策略模式）
│   └── api.py                 # FastAPI 后端
├── tests/
│   ├── __init__.py
│   ├── test_system.py         # 完整测试套件
│   └── README.md              # 测试说明
├── app.py                      # Streamlit 前端
├── main.py                     # 主入口
├── requirements.txt            # Python 依赖
├── Dockerfile                  # Docker 镜像配置
├── docker-compose.yml          # Docker Compose 配置
├── .env.example                # 环境变量示例
└── README.md                   # 项目说明文档
```

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
11. Add test suite - 添加完整测试套件
12. Add parent-child chunking - 添加父子切块功能
13. Add hybrid retrieval and rerank - 添加混合检索和 Rerank
14. Update prompts for better summarization - 优化提示词增强总结功能
15. Add streaming output and feedback system - 添加流式输出和用户反馈系统
16. Fix streaming output and improve status prompts - 修复流式输出并优化状态提示
17. Add streaming and feedback test script - 添加流式输出和反馈系统测试脚本

## 测试说明

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/test_system.py -v

# 运行流式输出和反馈系统测试
python -m pytest tests/test_streaming_and_feedback.py -v

# 运行测试并生成覆盖率报告
pip install pytest-cov
python -m pytest tests/test_system.py --cov=src --cov-report=html
```

### 测试覆盖范围

- **tests/test_system.py**: 完整系统测试（14 个测试）
  - 数据处理模块测试
  - 向量存储模块测试
  - 检索器模块测试
  - QA 生成器测试
  - 端到端集成测试

- **tests/test_streaming_and_feedback.py**: 新增功能测试（12 个测试）
  - 流式输出数据格式验证
  - 状态提示验证
  - 用户反馈数据结构验证
  - SSE 协议格式验证

## 许可证

本项目仅供学习和研究使用。
