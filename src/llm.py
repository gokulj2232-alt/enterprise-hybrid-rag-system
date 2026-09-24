import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
FALLBACK_ANSWER = (
    "I don't know based on the provided documents."
)
class LLM:
    def __init__(
        self,
        model_name=MODEL_NAME
    ):
        print(
            f"Loading LLM model: {model_name}",
            flush=True
        )
        print(
            "Loading tokenizer...",
            flush=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )
        print(
            "Tokenizer loaded.",
            flush=True
        )
        print(
            "Loading Qwen model weights...",
            flush=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        print(
            "Qwen model weights loaded.",
            flush=True
        )
        self.model.eval()
        print(
            "LLM loaded successfully!",
            flush=True
        )
    def generate(self, prompt):
        if not prompt or not prompt.strip():
            return FALLBACK_ANSWER
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict document question "
                    "answering assistant. "
                    "You MUST answer using ONLY the "
                    "information provided by the user. "
                    "Never use your general knowledge. "
                    "Never add facts that are not explicitly "
                    "supported by the provided context. "
                    "If the context does not contain enough "
                    "information, answer exactly: "
                    "I don't know based on the provided documents."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        formatted_prompt = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        generated_tokens = outputs[
            0
        ][
            inputs["input_ids"].shape[1]:
        ]
        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()
        if not answer:
            return FALLBACK_ANSWER
        unwanted_markers = [
            "[Source 1]",
            "[Source 2]",
            "[Source 3]",
            "[Source 4]",
            "[Source 5]",
            "(DOC001)",
            "(DOC002)"
        ]
        for marker in unwanted_markers:
            answer = answer.replace(
                marker,
                ""
            )
        answer = " ".join(
            answer.split()
        ).strip()
        if not answer:
            return FALLBACK_ANSWER
        return answer
if __name__ == "__main__":
    print("=" * 60)
    print("QWEN RAG LLM TEST")
    print("=" * 60)
    llm = LLM()
    test_prompt = """
CONTEXT:
Cloud computing is a technology topic.
This document explains cloud computing.
It provides information about the main concepts,
common processes, and practical considerations.
QUESTION:
What is cloud computing?
Answer using ONLY the context.
If the context does not explicitly define cloud
computing, answer:
I don't know based on the provided documents.
Give only one short answer.
"""
    answer = llm.generate(
        test_prompt
    )
    print("\n" + "=" * 60)
    print("GENERATED ANSWER")
    print("=" * 60)
    print(answer)
