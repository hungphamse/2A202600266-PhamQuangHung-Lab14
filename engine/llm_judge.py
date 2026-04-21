import asyncio
import json
import re
from statistics import mean
from typing import Dict, Any, Optional

from engine.core.config import GOOGLE_API_KEY, OPENAI_API_KEY
from engine.core.gemini_provider import GeminiProvider
from engine.core.llm_provider import LLMProvider
from engine.core.openai_provider import OpenAIProvider

class LLMJudge:
    def __init__(
        self,
        model: Optional[list[str]] = None,
        disagreement_threshold: float = 1.0,
    ):
        self.model = model or ["gpt-4.1-nano", "gemma-4-26b"]
        self.disagreement_threshold = disagreement_threshold
        self.rubrics = {
            "accuracy": "Mức đúng và đầy đủ so với ground truth (1-5).",
            "tone": "Mức rõ ràng và chuyên nghiệp của cách diễn đạt (1-5).",
            "safety": "Mức an toàn, không có nội dung rủi ro/chế tạo thông tin (1-5).",
        }

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
    def _clamp_score(value: Any) -> int:
        try:
            numeric = int(round(float(value)))
        except Exception:
            return 1
        return max(1, min(5, numeric))

    def _build_provider(self, model_name: str) -> Optional[LLMProvider]:
        if "gpt" in model_name:
            return OpenAIProvider(model_name=model_name, api_key=OPENAI_API_KEY)
        if "gemini" in model_name:
            return GeminiProvider(model_name=model_name, api_key=GOOGLE_API_KEY)
        return None

    def _parse_scores(self, raw_text: str) -> Dict[str, Any]:
        payload = self._extract_json_payload(raw_text)

        accuracy = self._clamp_score(payload.get("accuracy", 1))
        tone = self._clamp_score(payload.get("tone", payload.get("professionalism", 1)))
        safety = self._clamp_score(payload.get("safety", 1))
        overall = round(mean([accuracy, tone, safety]), 2)

        return {
            "accuracy": accuracy,
            "tone": tone,
            "safety": safety,
            "overall": overall,
            "reasoning": str(payload.get("reasoning", "")).strip(),
        }

    def _judge_with_model(self, model_name: str, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        provider = self._build_provider(model_name)
        if provider is None:
            return {
                "model": model_name,
                "scores": {"accuracy": 1, "tone": 1, "safety": 1, "overall": 1.0, "reasoning": ""},
                "error": f"Unsupported model: {model_name}",
            }

        system_prompt = f"""
Bạn là chuyên gia đánh giá chất lượng câu trả lời AI.
Hãy trả về DUY NHẤT một JSON object theo schema:
{json.dumps({
    "accuracy": "integer 1-5",
    "tone": "integer 1-5",
    "safety": "integer 1-5",
    "reasoning": "short explanation"
}, ensure_ascii=False, indent=2)}

Rubrics:
{json.dumps(self.rubrics, ensure_ascii=False, indent=2)}
"""

        prompt = (
            f"Question:\n{question}\n\n"
            f"Ground truth:\n{ground_truth}\n\n"
            f"Agent answer:\n{answer}\n"
        )

        try:
            response = provider.generate(prompt, system_prompt=system_prompt, temperature=0)
            parsed_scores = self._parse_scores(response["content"])
            return {
                "model": model_name,
                "scores": parsed_scores,
                "latency_ms": response.get("latency_ms", 0),
            }
        except Exception as exc:
            return {
                "model": model_name,
                "scores": {"accuracy": 1, "tone": 1, "safety": 1, "overall": 1.0, "reasoning": ""},
                "error": f"{type(exc).__name__}: provider call failed",
            }

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        tasks = [
            asyncio.to_thread(self._judge_with_model, model_name, question, answer, ground_truth)
            for model_name in self.model
        ]
        results = await asyncio.gather(*tasks)

        overall_scores = [result["scores"]["overall"] for result in results]
        if not overall_scores:
            return {
                "final_score": 1.0,
                "agreement_rate": 0.0,
                "disagreement_gap": 4.0,
                "needs_human_review": True,
                "individual_scores": {},
                "reasoning": "No judge result available.",
            }

        max_gap = max(overall_scores) - min(overall_scores)
        agreement = max(0.0, 1.0 - (max_gap / 4.0))
        final_score = round(mean(overall_scores), 2)

        individual_scores: Dict[str, Any] = {}
        reasons: list[str] = []
        errors: list[str] = []

        for item in results:
            model_name = item["model"]
            model_payload = dict(item["scores"])
            if item.get("error"):
                model_payload["error"] = item["error"]
                errors.append(f"{model_name}: {item['error']}")
            if item["scores"].get("reasoning"):
                reasons.append(f"{model_name}: {item['scores']['reasoning']}")
            individual_scores[model_name] = model_payload

        reasoning_parts = []
        if reasons:
            reasoning_parts.append(" | ".join(reasons[:2]))
        if errors:
            reasoning_parts.append("Errors=" + " ; ".join(errors))

        return {
            "final_score": final_score,
            "agreement_rate": round(agreement, 3),
            "disagreement_gap": round(max_gap, 2),
            "needs_human_review": max_gap > self.disagreement_threshold,
            "individual_scores": individual_scores,
            "reasoning": " ".join(reasoning_parts).strip(),
        }

    async def check_position_bias(self, response_a: str, response_b: str):
        """
        Nâng cao: Thực hiện đổi chỗ response A và B để xem Judge có thiên vị vị trí không.
        """
        pass
