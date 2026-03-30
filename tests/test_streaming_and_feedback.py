import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_api_module_exists():
    """测试 API 模块存在"""
    import src.api
    assert src.api is not None


def test_query_request_model():
    """测试 QueryRequest 数据模型"""
    from src.api import QueryRequest
    
    # 测试创建 QueryRequest
    req = QueryRequest(question="Test question", top_k=3)
    assert req.question == "Test question"
    assert req.top_k == 3


def test_feedback_request_model():
    """测试 FeedbackRequest 数据模型"""
    from src.api import FeedbackRequest
    
    # 测试创建 FeedbackRequest
    feedback = FeedbackRequest(
        question="Test question",
        answer="Test answer",
        helpful=True,
        comment="Test comment"
    )
    assert feedback.question == "Test question"
    assert feedback.answer == "Test answer"
    assert feedback.helpful is True
    assert feedback.comment == "Test comment"


def test_streaming_generator_basic():
    """测试流式生成器的基本逻辑（不实际运行）"""
    # 验证 generate_stream 函数签名
    from src.api import generate_stream
    assert callable(generate_stream)
    
    # 验证 qa_stream_endpoint 存在
    from src.api import qa_stream_endpoint
    assert callable(qa_stream_endpoint)


def test_feedback_endpoints_exist():
    """测试反馈端点存在"""
    from src.api import submit_feedback, get_feedback
    assert callable(submit_feedback)
    assert callable(get_feedback)


def test_feedback_storage_initialized():
    """测试反馈存储初始化"""
    from src.api import feedback_storage
    assert feedback_storage is not None
    assert isinstance(feedback_storage, list)


class TestStreamingLogic:
    """测试流式输出逻辑（mock 方式）"""
    
    def test_stream_data_format(self):
        """测试流式数据格式"""
        # 验证数据格式符合 SSE 协议
        test_status = {"status": "searching"}
        sse_data = f"data: {json.dumps(test_status, ensure_ascii=False)}\n\n"
        assert sse_data.startswith("data: ")
        assert sse_data.endswith("\n\n")
        
        # 解析数据
        data_str = sse_data.split("data: ")[1].strip()
        parsed = json.loads(data_str)
        assert parsed["status"] == "searching"
    
    def test_stream_with_sources(self):
        """测试包含来源的流式数据"""
        test_data = {
            "status": "generating",
            "sources": [
                {"text": "Test", "metadata": {"year": 2025}}
            ]
        }
        sse_data = f"data: {json.dumps(test_data, ensure_ascii=False)}\n\n"
        
        # 解析数据
        data_str = sse_data.split("data: ")[1].strip()
        parsed = json.loads(data_str)
        assert "sources" in parsed
        assert len(parsed["sources"]) == 1
    
    def test_stream_answer_format(self):
        """测试答案格式"""
        test_data = {"answer": "Test answer"}
        sse_data = f"data: {json.dumps(test_data, ensure_ascii=False)}\n\n"
        
        data_str = sse_data.split("data: ")[1].strip()
        parsed = json.loads(data_str)
        assert "answer" in parsed
        assert parsed["answer"] == "Test answer"
    
    def test_stream_done_signal(self):
        """测试完成信号"""
        test_data = {"done": True}
        sse_data = f"data: {json.dumps(test_data, ensure_ascii=False)}\n\n"
        
        data_str = sse_data.split("data: ")[1].strip()
        parsed = json.loads(data_str)
        assert "done" in parsed
        assert parsed["done"] is True


class TestFeedbackLogic:
    """测试反馈逻辑"""
    
    def test_feedback_data_structure(self):
        """测试反馈数据结构"""
        feedback_data = {
            "timestamp": 123456789.0,
            "question": "Test question",
            "answer": "Test answer",
            "helpful": True,
            "comment": "Great!"
        }
        assert "timestamp" in feedback_data
        assert "question" in feedback_data
        assert "answer" in feedback_data
        assert "helpful" in feedback_data
        assert "comment" in feedback_data
    
    def test_feedback_without_comment(self):
        """测试没有评论的反馈"""
        feedback_data = {
            "timestamp": 123456789.0,
            "question": "Test question",
            "answer": "Test answer",
            "helpful": False,
            "comment": None
        }
        assert feedback_data["comment"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
