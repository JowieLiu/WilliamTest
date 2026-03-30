import os
import sys
import tempfile
import shutil
import pytest
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processor import DataProcessor
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.qa_generator import (
    BaseQAGenerator,
    FallbackGenerator,
    TinyLlamaGenerator,
    OpenAIApiGenerator,
    set_global_generator,
    get_global_generator,
    get_generator_info,
    create_qa_generator
)


class TestDataProcessor:
    """测试数据处理器"""
    
    def test_data_processor_initialization(self):
        """测试 DataProcessor 初始化"""
        data_path = "data/aapl_10k.json"
        processor = DataProcessor(data_path)
        assert processor is not None
        assert processor.json_path == data_path
    
    def test_process_all(self):
        """测试处理所有文档"""
        data_path = "data/aapl_10k.json"
        processor = DataProcessor(data_path)
        chunks = processor.process_all()
        assert len(chunks) > 0
        for chunk in chunks:
            assert hasattr(chunk, 'text')
            assert hasattr(chunk, 'metadata')
            assert hasattr(chunk, 'year')


class TestVectorStore:
    """测试向量存储"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        try:
            shutil.rmtree(temp_path)
        except:
            pass
    
    def test_vector_store_initialization(self, temp_dir):
        """测试 VectorStore 初始化"""
        vector_store = VectorStore(temp_dir, "all-MiniLM-L6-v2")
        assert vector_store is not None
        assert vector_store.client is not None
        assert vector_store.embedding_model is not None
    
    def test_create_collection(self, temp_dir):
        """测试创建集合"""
        vector_store = VectorStore(temp_dir, "all-MiniLM-L6-v2")
        vector_store.create_collection()
        assert vector_store.collection is not None
    
    def test_add_documents(self, temp_dir):
        """测试添加文档"""
        from src.data_processor import DocumentChunk
        
        vector_store = VectorStore(temp_dir, "all-MiniLM-L6-v2")
        vector_store.create_collection()
        
        test_chunks = [
            DocumentChunk(
                id="chunk_0",
                text="Apple Inc. is an American technology company.",
                metadata={"year": 2025, "section_title": "Overview"},
                year=2025,
                section_title="Overview"
            )
        ]
        vector_store.add_documents(test_chunks)
        # 验证文档已添加（通过检查集合数量）
        assert vector_store.collection.count() >= 1


class TestRetriever:
    """测试检索器"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        try:
            shutil.rmtree(temp_path)
        except:
            pass
    
    @pytest.fixture
    def vector_store(self, temp_dir):
        """创建测试向量存储"""
        from src.data_processor import DocumentChunk
        
        vector_store = VectorStore(temp_dir, "all-MiniLM-L6-v2")
        vector_store.create_collection()
        
        test_chunks = [
            DocumentChunk(
                id="chunk_0",
                text="Apple's revenue in 2025 was $394 billion.",
                metadata={"year": 2025, "section_title": "Financial Results"},
                year=2025,
                section_title="Financial Results"
            ),
            DocumentChunk(
                id="chunk_1",
                text="Apple's main products include iPhone, Mac, iPad, and Apple Watch.",
                metadata={"year": 2025, "section_title": "Products"},
                year=2025,
                section_title="Products"
            )
        ]
        vector_store.add_documents(test_chunks)
        return vector_store
    
    def test_retriever_initialization(self, vector_store):
        """测试 Retriever 初始化"""
        retriever = Retriever(vector_store)
        assert retriever is not None
        assert retriever.vector_store == vector_store
    
    def test_retrieve(self, vector_store):
        """测试检索功能"""
        retriever = Retriever(vector_store)
        results = retriever.retrieve("Apple revenue", top_k=2)
        assert len(results) > 0
        for result in results:
            assert 'text' in result
            assert 'metadata' in result
    
    def test_format_context(self, vector_store):
        """测试上下文格式化"""
        retriever = Retriever(vector_store)
        results = retriever.retrieve("Apple products", top_k=2)
        context = retriever.format_context(results)
        assert isinstance(context, str)
        assert len(context) > 0


class TestQAGenerators:
    """测试 QA 生成器"""
    
    def test_fallback_generator(self):
        """测试 FallbackGenerator"""
        generator = FallbackGenerator()
        assert generator.is_available() is True
        
        query = "What is Apple's revenue?"
        context = "Apple's revenue in 2025 was $394 billion."
        answer = generator.generate_answer(query, context)
        assert isinstance(answer, str)
        assert len(answer) > 0
    
    def test_tinyllama_generator_available(self):
        """测试 TinyLlamaGenerator（检查是否可用）"""
        try:
            generator = TinyLlamaGenerator()
            # 即使模型没加载，也应该返回一个生成器对象
            assert generator is not None
        except Exception as e:
            pytest.skip(f"TinyLlama model not available: {e}")
    
    @patch('src.qa_generator.requests.post')
    def test_openai_api_generator(self, mock_post):
        """测试 OpenAIApiGenerator"""
        # 模拟 API 响应
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Apple's revenue was $394 billion."}}]
        }
        mock_post.return_value = mock_response
        
        generator = OpenAIApiGenerator(
            api_url="https://api.openai.com/v1",
            api_key="test-key",
            model_name="gpt-3.5-turbo"
        )
        assert generator.is_available() is True
        
        query = "What is Apple's revenue?"
        context = "Apple's revenue in 2025 was $394 billion."
        answer = generator.generate_answer(query, context)
        assert isinstance(answer, str)
    
    def test_global_generator_management(self):
        """测试全局生成器管理"""
        # 测试设置和获取全局生成器
        generator1 = FallbackGenerator()
        set_global_generator(generator1)
        assert get_global_generator() == generator1
        
        # 测试获取生成器信息
        info = get_generator_info()
        assert 'type' in info
        assert 'name' in info
        assert 'description' in info
    
    def test_create_qa_generator(self):
        """测试 create_qa_generator 工厂函数"""
        # 测试 fallback 模式
        generator = create_qa_generator("fallback")
        assert isinstance(generator, FallbackGenerator)
        
        # 测试 tinyllama 模式（如果可用）
        try:
            generator = create_qa_generator("tinyllama")
            # 即使加载失败，也应该返回 FallbackGenerator
            assert isinstance(generator, (TinyLlamaGenerator, FallbackGenerator))
        except Exception:
            pass


class TestEndToEnd:
    """端到端集成测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        try:
            shutil.rmtree(temp_path)
        except:
            pass
    
    def test_full_workflow(self, temp_dir):
        """测试完整工作流程"""
        # 1. 加载和处理数据
        data_path = "data/aapl_10k.json"
        processor = DataProcessor(data_path)
        chunks = processor.process_all()
        assert len(chunks) > 0
        
        # 2. 创建向量存储
        vector_store = VectorStore(temp_dir, "all-MiniLM-L6-v2")
        vector_store.create_collection()
        vector_store.add_documents(chunks[:10])  # 只添加前10个块进行快速测试
        
        # 3. 创建检索器
        retriever = Retriever(vector_store)
        
        # 4. 检索
        results = retriever.retrieve("Apple products", top_k=3)
        assert len(results) > 0
        
        # 5. 格式化上下文
        context = retriever.format_context(results)
        
        # 6. 生成答案
        generator = FallbackGenerator()
        answer = generator.generate_answer("What are Apple's products?", context)
        assert isinstance(answer, str)
        assert len(answer) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
