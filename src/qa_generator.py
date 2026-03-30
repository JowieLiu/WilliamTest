from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


class QAGenerator:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model {model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device if self.device == "cuda" else -1
        )
        
        print("Model loaded successfully!")

    def generate_answer(self, query: str, context: str) -> str:
        prompt = self._build_prompt(query, context)
        
        try:
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
            
            if len(answer) < 10 or len(answer.split()) < 3:
                return self._fallback_answer(context)
            
            return answer
        except Exception as e:
            print(f"Generation error: {e}")
            return self._fallback_answer(context)

    def _fallback_answer(self, context: str) -> str:
        lines = context.split('\n')
        relevant_parts = []
        for line in lines[:5]:
            if line.strip() and len(line.strip()) > 20:
                relevant_parts.append(line.strip())
        
        if relevant_parts:
            return "根据检索到的财报信息：\n" + "\n".join(relevant_parts[:3])
        return "根据检索到的财报信息，相关内容已在参考来源中展示。"

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""You are a helpful financial analyst. Use the following context to answer the question.

Context:
{context}

Question: {query}

Answer:"""
