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
        
        outputs = self.pipe(
            prompt,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            return_full_text=False
        )
        
        answer = outputs[0]['generated_text'].strip()
        return answer

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""<|system|>
You are a financial analyst. Answer the question based on the provided context. If you don't know the answer, just say you don't know.
<|user|>
Context: {context}

Question: {query}
<|assistant|>"""
