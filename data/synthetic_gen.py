import json
import asyncio
import textwrap

from engine.core.config import OPENAI_API_KEY
from engine.core.openai_provider import OpenAIProvider

# Giả lập việc gọi LLM để tạo dữ liệu (Students will implement this)
async def generate_qa_from_text(text: str, num_pairs: int = 5) -> list[dict]:
    """
    TODO: Sử dụng OpenAI/Anthropic API để tạo các cặp (Question, Expected Answer, Context)
    từ đoạn văn bản cho trước.
    Yêu cầu: Tạo ít nhất 1 câu hỏi 'lừa' (adversarial) hoặc cực khó.
    """
    print(f"Generating {num_pairs} QA pairs from text...")
    qas: list[dict] = []
    
    provider = OpenAIProvider(model_name="gpt-4o-mini", api_key=OPENAI_API_KEY)
    prompt = textwrap.dedent(f"""
    Bạn là một chuyên gia tạo dữ liệu câu hỏi cho việc đánh giá AI. 
    Hãy đọc đoạn văn bản sau và tạo ra {num_pairs} cặp câu hỏi và câu trả lời kỳ vọng dựa trên văn bản được cung cấp. Đảm bảo rằng ít nhất 1 trong số đó là câu hỏi 'lừa' hoặc không liên quan đến bối cảnh văn bản, nhằm thử thách khả năng hiểu biết của AI. Trả về kết quả dưới dạng JSON như sau:
    [
        {{
            "question": "Câu hỏi mẫu từ tài liệu?",
            "expected_answer": "Câu trả lời kỳ vọng mẫu.",
            "context": "Giải thích ngắn gọn về lý do tại sao câu hỏi này được tạo ra và nó liên quan đến văn bản như thế nào.",
            "metadata": {
                "difficulty": <"easy" | "medium" | "hard" >, 
                "type": <"fact-check" | "adversarial" | "out-of-context">
            }
        }}
        ... (Các bộ QA khác) ...
    ]
    Đoạn văn bản:
    {text}
    """)

    response = provider.generate(prompt, temperature=1)
    try:
        qas = json.loads(response['content'])
        return qas
    except Exception as e:
        print(f"Error parsing generated QA pairs: {e}")
        return []

async def main():
    with open("data/docs/access_control_sop.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    qa_pairs = await generate_qa_from_text(raw_text, num_pairs=10)
    
    with open("data/golden_set.jsonl", "w", encoding="utf-8") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print("Done! Saved to data/golden_set.jsonl")

if __name__ == "__main__":
    asyncio.run(main())
