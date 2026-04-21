import asyncio
import re
from pathlib import Path
from typing import Dict, Any, Optional

from engine.core.config import GOOGLE_API_KEY, OPENAI_API_KEY
from engine.core.gemini_provider import GeminiProvider
from engine.core.llm_provider import LLMProvider
from engine.core.openai_provider import OpenAIProvider

class MainAgent:
    def __init__(self, docs_dir: str = "data/docs", top_k: int = 3, model_name: Optional[str] = None):
        self.name = "SupportAgent-v1"
        self.top_k = top_k
        self.docs_dir = Path(docs_dir)
        self.documents = self._load_documents()
        self.providers = self._build_providers(model_name)
        self.model_name = model_name or (self.providers[0].model_name if self.providers else "local-fallback")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    def _load_documents(self) -> list[Dict[str, Any]]:
        if not self.docs_dir.exists():
            return []

        docs: list[Dict[str, Any]] = []
        for file_path in sorted(self.docs_dir.glob("*.txt")):
            content = file_path.read_text(encoding="utf-8")
            docs.append(
                {
                    "id": file_path.stem.lower(),
                    "text": content,
                    "tokens": set(self._tokenize(content)),
                }
            )
        return docs

    def _build_providers(self, preferred_model: Optional[str]) -> list[LLMProvider]:
        providers: list[LLMProvider] = []

        if preferred_model and "gpt" in preferred_model and OPENAI_API_KEY:
            providers.append(OpenAIProvider(model_name=preferred_model, api_key=OPENAI_API_KEY))
        elif preferred_model and "gemini" in preferred_model and GOOGLE_API_KEY:
            providers.append(GeminiProvider(model_name=preferred_model, api_key=GOOGLE_API_KEY))

        if OPENAI_API_KEY and not any(isinstance(provider, OpenAIProvider) for provider in providers):
            providers.append(OpenAIProvider(model_name="gpt-4o-mini", api_key=OPENAI_API_KEY))
        if GOOGLE_API_KEY and not any(isinstance(provider, GeminiProvider) for provider in providers):
            providers.append(GeminiProvider(model_name="gemini-2.5-flash", api_key=GOOGLE_API_KEY))
        return providers

    @staticmethod
    def _build_local_fallback_answer(contexts: list[Dict[str, Any]], error_message: str = "") -> str:
        base = (
            "Khong goi duoc provider LLM, day la tom tat tu context da truy xuat:\n"
            + "\n".join(f"- ({ctx['id']}) {ctx['text'][:180].strip()}" for ctx in contexts)
        )
        if error_message:
            return f"{base}\n(LLM error: provider unavailable)"
        return base

    def _retrieve(self, question: str) -> list[Dict[str, Any]]:
        if not self.documents:
            return []

        question_tokens = set(self._tokenize(question))
        if not question_tokens:
            return []

        scored_docs: list[Dict[str, Any]] = []
        for doc in self.documents:
            overlap = len(question_tokens.intersection(doc["tokens"]))
            score = overlap / len(question_tokens)
            scored_docs.append({"id": doc["id"], "text": doc["text"], "score": round(score, 4)})

        scored_docs.sort(key=lambda item: item["score"], reverse=True)
        top_docs = [item for item in scored_docs[: self.top_k] if item["score"] > 0]
        return top_docs or scored_docs[: self.top_k]

    async def _generate_answer(self, question: str, contexts: list[Dict[str, Any]]) -> str:
        if not contexts:
            return "Tôi chưa tìm thấy tài liệu phù hợp để trả lời câu hỏi này."

        context_block = "\n\n".join(
            f"[source={ctx['id']}]\n{ctx['text'][:1400]}" for ctx in contexts
        )

        if not self.providers:
            return self._build_local_fallback_answer(contexts)

        system_prompt = (
            "Bạn là trợ lý hỗ trợ nội bộ. Chỉ trả lời dựa trên context được cung cấp. "
            "Nếu không đủ dữ liệu thì nói rõ không đủ dữ liệu, không được bịa thêm."
        )
        prompt = (
            f"Question:\n{question}\n\n"
            f"Context documents:\n{context_block}\n\n"
            "Trả lời ngắn gọn, chính xác, có thể trích source id nếu cần."
        )

        last_error = ""
        for provider in self.providers:
            try:
                response = await asyncio.to_thread(
                    provider.generate,
                    prompt,
                    system_prompt,
                    0.2,
                )
                content = str(response.get("content", "")).strip()
                if content:
                    self.model_name = provider.model_name
                    return content
            except Exception as exc:
                last_error = str(exc)

        return self._build_local_fallback_answer(contexts, last_error)

    async def query(self, question: str) -> Dict:
        retrieved = self._retrieve(question)
        answer = await self._generate_answer(question, retrieved)

        contexts = [doc["text"][:1200] for doc in retrieved]
        retrieved_ids = [doc["id"] for doc in retrieved]

        return {
            "answer": answer,
            "contexts": contexts,
            "retrieved_ids": retrieved_ids,
            "metadata": {
                "model": self.model_name,
                "sources": retrieved_ids,
                "retrieval_scores": {doc["id"]: doc["score"] for doc in retrieved},
            },
        }

if __name__ == "__main__":
    agent = MainAgent()
    async def test():
        resp = await agent.query("Làm thế nào để đổi mật khẩu?")
        print(resp)
    asyncio.run(test())
