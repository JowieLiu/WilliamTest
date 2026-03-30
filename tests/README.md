# AAPL 10-K QA 系统测试

本目录包含系统的完整测试套件。

## 测试覆盖范围

### 1. 数据处理测试 (`TestDataProcessor`)
- `test_data_processor_initialization`: 测试 DataProcessor 初始化
- `test_process_all`: 测试处理所有文档

### 2. 向量存储测试 (`TestVectorStore`)
- `test_vector_store_initialization`: 测试 VectorStore 初始化
- `test_create_collection`: 测试创建集合
- `test_add_documents`: 测试添加文档

### 3. 检索器测试 (`TestRetriever`)
- `test_retriever_initialization`: 测试 Retriever 初始化
- `test_retrieve`: 测试检索功能
- `test_format_context`: 测试上下文格式化

### 4. QA 生成器测试 (`TestQAGenerators`)
- `test_fallback_generator`: 测试 FallbackGenerator
- `test_tinyllama_generator_available`: 测试 TinyLlamaGenerator 可用性
- `test_openai_api_generator`: 测试 OpenAIApiGenerator（使用 mock）
- `test_global_generator_management`: 测试全局生成器管理
- `test_create_qa_generator`: 测试 create_qa_generator 工厂函数

### 5. 端到端集成测试 (`TestEndToEnd`)
- `test_full_workflow`: 测试完整工作流程

## 运行测试

### 前置条件
确保已激活 conda 环境：
```bash
conda activate aapl-qa-env
```

### 运行所有测试
```bash
python -m pytest tests/test_system.py -v
```

### 运行特定测试类
```bash
python -m pytest tests/test_system.py::TestDataProcessor -v
```

### 运行特定测试方法
```bash
python -m pytest tests/test_system.py::TestQAGenerators::test_fallback_generator -v
```

### 生成覆盖率报告
```bash
pip install pytest-cov
python -m pytest tests/test_system.py --cov=src --cov-report=html
```

## 测试结果

所有测试已通过 (14/14 passed)！
