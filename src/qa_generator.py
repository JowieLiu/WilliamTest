from abc import ABC, abstractmethod
from typing import Optional
import os
import requests


class BaseQAGenerator(ABC):
    @abstractmethod
    def generate_answer(self, query: str, context: str) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass


class FallbackGenerator(BaseQAGenerator):
    def generate_answer(self, query: str, context: str) -> str:
        lines = context.split('\n')
        relevant_parts = []
        for line in lines[:8]:
            if line.strip() and len(line.strip()) > 20:
                relevant_parts.append(line.strip())
        
        if relevant_parts:
            return "根据检索到的财报信息：\n\n" + "\n".join(relevant_parts[:5])
        return "根据检索到的财报信息，相关内容已在参考来源中展示。"

    def is_available(self) -> bool:
        return True


class TinyLlamaGenerator(BaseQAGenerator):
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self._loaded = False
        self._load_model()

    def _load_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading TinyLlama model {self.model_name} on {self.device}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device if self.device == "cuda" else -1
            )
            
            self._loaded = True
            print("TinyLlama model loaded successfully!")
        except Exception as e:
            print(f"Failed to load TinyLlama: {e}")
            self._loaded = False

    def generate_answer(self, query: str, context: str) -> str:
        if not self._loaded:
            return FallbackGenerator().generate_answer(query, context)
        
        try:
            prompt = self._build_prompt(query, context)
            
            outputs = self.pipe(
                prompt,
                max_new_tokens=384,
                temperature=0.3,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.1,
                do_sample=True,
                return_full_text=False
            )
            
            answer = outputs[0]['generated_text'].strip()
            
            if len(answer) < 10 or len(answer.split()) < 3 or "MD MD" in answer:
                return FallbackGenerator().generate_answer(query, context)
            
            return answer
        except Exception as e:
            print(f"Generation error: {e}")
            return FallbackGenerator().generate_answer(query, context)

    def is_available(self) -> bool:
        return self._loaded

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""<|system|>
You are a professional financial analyst. Your task is to answer the user's question based ONLY on the provided context from Apple's 10-K financial reports.

Key instructions:
1. Answer the question using only information from the context
2. If the context doesn't contain enough information, say so clearly
3. Be concise and direct
4. Use financial terminology appropriately
5. Focus on factual information from the reports</s>

<|user|>
Context information from Apple 10-K reports:
{context}

Question: {query}</s>

<|assistant|>
Based on the provided context, here is the answer:"""


class OpenAIApiGenerator(BaseQAGenerator):
    def __init__(self, api_url: str, api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name
        self._available = bool(api_url and api_key)

    def generate_answer(self, query: str, context: str) -> str:
        if not self._available:
            return FallbackGenerator().generate_answer(query, context)
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": """You are a professional financial analyst specializing in Apple Inc.'s 10-K reports. Your task is to provide accurate, concise answers based exclusively on the provided context.

## Guidelines:
1. **Only use information from the provided context** - do not use external knowledge
2. **Be direct and factual** - avoid speculation or opinions
3. **If information is not available**, clearly state "The provided context does not contain sufficient information to answer this question."
4. **Use financial terminology appropriately**
5. **Cite relevant sections** when possible
6. **Keep answers focused** on the specific question asked
7. **Maintain professional tone** suitable for financial analysis
8. **Summarize the key points** from the context that answer the question"""
                },
                {
                    "role": "user",
                    "content": f"""## Context from Apple 10-K Reports:
{context}

## Question:
{query}

Please provide a comprehensive summary and answer based solely on the context provided above."""
                }
            ]
            
            print(f"Calling API at: {self.api_url}/chat/completions")
            print(f"Using model: {self.model_name}")
            
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 1024
                },
                timeout=60
            )
            
            print(f"API Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"].strip()
                print(f"Generated answer length: {len(answer)}")
                return answer
            else:
                error_msg = f"API Error: {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f" - {error_detail}"
                except:
                    error_msg += f" - {response.text[:200]}"
                print(error_msg)
                return f"⚠️ API 调用失败 ({response.status_code})，使用本地模式展示检索结果。\n\n" + FallbackGenerator().generate_answer(query, context)
                
        except Exception as e:
            print(f"API Generation error: {e}")
            return f"⚠️ API 调用出错: {str(e)}，使用本地模式展示检索结果。\n\n" + FallbackGenerator().generate_answer(query, context)

    def is_available(self) -> bool:
        return self._available


_global_generator: Optional[BaseQAGenerator] = None


def set_global_generator(generator: BaseQAGenerator):
    global _global_generator
    _global_generator = generator


def get_global_generator() -> BaseQAGenerator:
    global _global_generator
    if _global_generator is None:
        return FallbackGenerator()
    return _global_generator


def get_generator_info() -> dict:
    global _global_generator
    if _global_generator is None:
        return {
            "type": "fallback",
            "name": "本地 Fallback 模式",
            "description": "直接展示检索结果"
        }
    
    if isinstance(_global_generator, TinyLlamaGenerator):
        return {
            "type": "tinyllama",
            "name": "TinyLlama 本地小模型",
            "description": "使用本地 TinyLlama-1.1B 模型进行总结"
        }
    elif isinstance(_global_generator, OpenAIApiGenerator):
        return {
            "type": "api",
            "name": "API 大模型",
            "description": f"使用 {_global_generator.model_name} 模型",
            "api_url": _global_generator.api_url,
            "model_name": _global_generator.model_name
        }
    else:
        return {
            "type": "fallback",
            "name": "本地 Fallback 模式",
            "description": "直接展示检索结果"
        }


def create_qa_generator(model_type: Optional[str] = None) -> BaseQAGenerator:
    model_type = model_type or os.getenv("GENERATION_MODEL_TYPE", "fallback")
    
    if model_type == "tinyllama":
        generator = TinyLlamaGenerator()
        if generator.is_available():
            return generator
    
    return FallbackGenerator()
