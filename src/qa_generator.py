from abc import ABC, abstractmethod
from typing import Optional
import os


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
        return f"""You are a helpful financial analyst. Use the following context to answer the question.

Context:
{context}

Question: {query}

Answer:"""


def create_qa_generator(model_type: Optional[str] = None) -> BaseQAGenerator:
    model_type = model_type or os.getenv("GENERATION_MODEL_TYPE", "fallback")
    
    if model_type == "tinyllama":
        generator = TinyLlamaGenerator()
        if generator.is_available():
            return generator
    
    return FallbackGenerator()
