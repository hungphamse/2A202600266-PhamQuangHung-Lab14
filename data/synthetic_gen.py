import json
import asyncio
import textwrap
import re
from pathlib import Path
from typing import Optional

from engine.core.config import GOOGLE_API_KEY, OPENAI_API_KEY
from engine.core.gemini_provider import GeminiProvider
from engine.core.llm_provider import LLMProvider
from engine.core.openai_provider import OpenAIProvider

# Gọi LLM để tạo dữ liệu
def _extract_json_array(raw_text: str) -> list[dict]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned).strip()
        cleaned = cleaned.removesuffix("```").strip()

    try:
        payload = json.loads(cleaned)
    except Exception:
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if not match:
            raise
        payload = json.loads(match.group(0))

    if not isinstance(payload, list):
        raise ValueError("LLM output is not a JSON array")
    return [item for item in payload if isinstance(item, dict)]


async def generate_qa_from_text(
    text: str,
    source_doc_id: str,
    provider: Optional[LLMProvider],
    num_pairs: int = 5,
) -> list[dict]:
    print(f"Generating {num_pairs} QA pairs from {source_doc_id}...")
    prompt = textwrap.dedent(f"""
    Bạn là một chuyên gia tạo dữ liệu câu hỏi cho việc đánh giá AI. 
    Hãy đọc đoạn văn bản dưới đây và tạo ra {num_pairs} cặp câu hỏi và câu trả lời kỳ vọng dựa trên văn bản được cung cấp. 
    Đảm bảo rằng ít nhất 1 trong số đó là câu hỏi 'lừa' (adversarial) hoặc không liên quan đến văn bản, hoặc những câu hỏi tiêm nhiễm, nhằm thử thách khả năng xử lý của AI.
    Trả về kết quả dưới dạng JSON như sau:
    [
        {{
            "question": "Câu hỏi liên quan đến văn bản, hoặc là câu hỏi 'lừa' hoặc không liên quan.",
            "expected_answer": "Câu trả lời kỳ vọng mẫu.",
            "context": "Giải thích ngắn gọn về lý do tại sao câu hỏi này được tạo ra và nó liên quan đến văn bản như thế nào.",
            "metadata": {{
                "difficulty": <"easy" | "medium" | "hard" >, 
                "type": <"fact-check" | "adversarial" | "out-of-context">
            }}
        }}
        ... (Các bộ QA khác) ...
    ]
    Đoạn văn bản:
    {text}
    """)

    if provider is None:
        return _fallback_generate_qa(text, source_doc_id, num_pairs)

    try:
        response = await asyncio.to_thread(provider.generate, prompt, None, 1)
        qas = _extract_json_array(str(response["content"]))
        for pair in qas:
            pair["expected_retrieval_ids"] = [source_doc_id]
            pair.setdefault("metadata", {})["source_doc_id"] = source_doc_id
        return qas
    except Exception as e:
        print(f"LLM generation failed for {source_doc_id}; switching to fallback QA generation. Exception: {e}")
        return _fallback_generate_qa(text, source_doc_id, num_pairs)


def _fallback_generate_qa(text: str, source_doc_id: str, num_pairs: int) -> list[dict]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    content_lines = [line for line in lines if not line.startswith("===")]
    if not content_lines:
        content_lines = [text[:300].strip()] if text.strip() else ["No content"]

    qas: list[dict] = []
    for idx in range(num_pairs):
        snippet = content_lines[idx % len(content_lines)]
        qas.append(
            {
                "question": f"Theo tai lieu {source_doc_id}, noi dung nao lien quan den: '{snippet[:80]}'?",
                "expected_answer": snippet,
                "context": f"Generated fallback QA from source {source_doc_id}.",
                "metadata": {
                    "difficulty": "medium" if idx % 3 else "hard",
                    "type": "fact-check" if idx else "adversarial",
                    "source_doc_id": source_doc_id,
                },
                "expected_retrieval_ids": [source_doc_id],
            }
        )
    return qas


async def main():
    provider: Optional[LLMProvider] = None
    if OPENAI_API_KEY:
        provider = OpenAIProvider(model_name="gpt-4.1-nano", api_key=OPENAI_API_KEY)
    elif GOOGLE_API_KEY:
        provider = GeminiProvider(model_name="gemini-2.5-flash", api_key=GOOGLE_API_KEY)
    else:
        print("⚠️ Warning: No LLM provider available. The generated QA pairs may be low quality. Please set OPENAI_API_KEY or GOOGLE_API_KEY for better results.")
    docs_dir = Path("data/docs")
    doc_files = sorted(docs_dir.glob("*.txt"))
    if not doc_files:
        raise RuntimeError("Không tìm thấy tài liệu trong data/docs")

    all_pairs: list[dict] = []
    for file_path in doc_files:
        raw_text = file_path.read_text(encoding="utf-8")
        pairs = await generate_qa_from_text(
            text=raw_text,
            source_doc_id=file_path.stem.lower(),
            provider=provider,
            num_pairs=10,
        )
        all_pairs.extend(pairs)

    with open("data/golden_set.jsonl", "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Done! Saved {len(all_pairs)} cases to data/golden_set.jsonl")

if __name__ == "__main__":
    asyncio.run(main())
