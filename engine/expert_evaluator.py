import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from engine.core.config import GOOGLE_API_KEY, OPENAI_API_KEY
from engine.core.gemini_provider import GeminiProvider
from engine.core.llm_provider import LLMProvider
from engine.core.openai_provider import OpenAIProvider
from engine.retrieval_eval import RetrievalEvaluator


class ExpertEvaluator:
    def __init__(self, model_name: Optional[str] = None, top_k: int = 3):
        self.retrieval_evaluator = RetrievalEvaluator(default_top_k=top_k)
        self.provider = self._build_provider(model_name)
        self.model_name = model_name or (self.provider.model_name if self.provider else "local-heuristic")
        self.known_doc_ids = self._load_doc_ids()

    @staticmethod
    def _load_doc_ids() -> list[str]:
        docs_dir = Path("data/docs")
        if not docs_dir.exists():
            return []
        return sorted(path.stem.lower() for path in docs_dir.glob("*.txt"))

    @staticmethod
    def _build_provider(preferred_model: Optional[str]) -> Optional[LLMProvider]:
        if preferred_model and "gpt" in preferred_model and OPENAI_API_KEY:
            return OpenAIProvider(model_name=preferred_model, api_key=OPENAI_API_KEY)
        if preferred_model and "gemini" in preferred_model and GOOGLE_API_KEY:
            return GeminiProvider(model_name=preferred_model, api_key=GOOGLE_API_KEY)

        if OPENAI_API_KEY:
            return OpenAIProvider(model_name="gpt-4.1-nano", api_key=OPENAI_API_KEY)
        if GOOGLE_API_KEY:
            return GeminiProvider(model_name="gemini-2.5-flash", api_key=GOOGLE_API_KEY)
        return None

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))

    def _extract_expected_ids(self, case: Dict[str, Any]) -> list[str]:
        expected_ids = case.get("expected_retrieval_ids") or []
        if isinstance(expected_ids, str):
            expected_ids = [expected_ids]

        metadata = case.get("metadata", {})
        for key in ("source_doc_id", "source_id", "doc_id"):
            if metadata.get(key):
                expected_ids.append(str(metadata[key]))
            if case.get(key):
                expected_ids.append(str(case[key]))

        if expected_ids:
            return [doc_id.lower().strip() for doc_id in expected_ids if str(doc_id).strip()]

        # Fallback: suy luận sơ bộ từ question theo từ khoi trong doc id.
        question = str(case.get("question", "")).lower()
        inferred: list[str] = []
        for doc_id in self.known_doc_ids:
            keywords = [part for part in doc_id.split("_") if len(part) > 2]
            if any(keyword in question for keyword in keywords):
                inferred.append(doc_id)
        return inferred[:1]

    @staticmethod
    def _extract_json_payload(text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned).strip()
            cleaned = cleaned.removesuffix("```").strip()

        try:
            return json.loads(cleaned)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if not match:
                raise
            return json.loads(match.group(0))

    @staticmethod
    def _clamp_unit_score(value: Any) -> float:
        try:
            numeric = float(value)
        except Exception:
            return 0.0
        return round(max(0.0, min(1.0, numeric)), 3)

    @classmethod
    def _overlap_score(cls, source_text: str, target_text: str) -> float:
        source_tokens = cls._tokenize(source_text)
        target_tokens = cls._tokenize(target_text)
        if not source_tokens or not target_tokens:
            return 0.0
        return len(source_tokens.intersection(target_tokens)) / len(source_tokens)

    def _heuristic_quality(self, case: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
        answer = str(response.get("answer", ""))
        expected_answer = str(case.get("expected_answer", ""))
        question = str(case.get("question", ""))
        contexts = " ".join(response.get("contexts", []))

        relevancy = self._overlap_score(answer, f"{question} {expected_answer}")
        faithfulness = self._overlap_score(answer, contexts)

        return {
            "faithfulness": round(faithfulness, 3),
            "relevancy": round(relevancy, 3),
            "reasoning": "Heuristic overlap score (no LLM evaluator configured).",
        }

    async def _llm_quality_score(self, case: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
        if not self.provider:
            return self._heuristic_quality(case, response)

        question = str(case.get("question", ""))
        expected_answer = str(case.get("expected_answer", ""))
        answer = str(response.get("answer", ""))
        contexts = "\n\n".join(response.get("contexts", []))

        system_prompt = (
            "Bạn là chuyên gia đánh giá output RAG. "
            "Trả về DUY NHAT mot JSON object voi 3 truong: faithfulness, relevancy, reasoning. "
            "faithfulness va relevancy la so thuc trong [0,1]."
        )
        prompt = (
            f"Question:\n{question}\n\n"
            f"Expected answer:\n{expected_answer}\n\n"
            f"Agent answer:\n{answer}\n\n"
            f"Retrieved contexts:\n{contexts}\n\n"
            "Danh gia chat luong va tra ve JSON hop le."
        )

        try:
            result = await asyncio.to_thread(
                self.provider.generate,
                prompt,
                system_prompt,
                0,
            )
            payload = self._extract_json_payload(str(result.get("content", "")))
            return {
                "faithfulness": self._clamp_unit_score(payload.get("faithfulness", 0.0)),
                "relevancy": self._clamp_unit_score(payload.get("relevancy", 0.0)),
                "reasoning": str(payload.get("reasoning", "")).strip(),
            }
        except Exception as exc:
            fallback = self._heuristic_quality(case, response)
            fallback["reasoning"] += f" LLM evaluator error: {exc}"
            return fallback

    async def score(self, case: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
        expected_ids = self._extract_expected_ids(case)
        retrieved_ids = response.get("retrieved_ids") or response.get("metadata", {}).get("sources", [])
        retrieval = self.retrieval_evaluator.evaluate_single(expected_ids, retrieved_ids)

        quality = await self._llm_quality_score(case, response)
        return {
            "faithfulness": quality["faithfulness"],
            "relevancy": quality["relevancy"],
            "retrieval": retrieval,
            "reasoning": quality["reasoning"],
            "evaluator_model": self.model_name,
        }
