from typing import List, Dict, Any

class RetrievalEvaluator:
    def __init__(self, default_top_k: int = 3):
        self.default_top_k = default_top_k

    @staticmethod
    def _normalize_ids(ids: List[str]) -> List[str]:
        return [str(doc_id).strip().lower() for doc_id in ids if str(doc_id).strip()]

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """Tính Hit Rate@k cho 1 test case."""
        normalized_expected = set(self._normalize_ids(expected_ids))
        normalized_retrieved = self._normalize_ids(retrieved_ids)
        if not normalized_expected:
            return 0.0

        top_retrieved = normalized_retrieved[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in normalized_expected)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """Tính Reciprocal Rank cho 1 test case."""
        normalized_expected = set(self._normalize_ids(expected_ids))
        normalized_retrieved = self._normalize_ids(retrieved_ids)
        if not normalized_expected:
            return 0.0

        for i, doc_id in enumerate(normalized_retrieved):
            if doc_id in normalized_expected:
                return 1.0 / (i + 1)
        return 0.0

    def evaluate_single(
        self,
        expected_ids: List[str],
        retrieved_ids: List[str],
        top_k: int | None = None,
    ) -> Dict[str, Any]:
        k = top_k or self.default_top_k
        normalized_expected = self._normalize_ids(expected_ids)
        normalized_retrieved = self._normalize_ids(retrieved_ids)

        return {
            "hit_rate": self.calculate_hit_rate(normalized_expected, normalized_retrieved, top_k=k),
            "mrr": self.calculate_mrr(normalized_expected, normalized_retrieved),
            "top_k": k,
            "expected_ids": normalized_expected,
            "retrieved_ids": normalized_retrieved,
            "available": bool(normalized_expected),
        }

    async def evaluate_batch(self, dataset: List[Dict]) -> Dict:
        """Tính trung bình retrieval metrics cho toàn bộ dataset."""
        if not dataset:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0, "evaluated_cases": 0, "coverage": 0.0}

        total_hit = 0.0
        total_mrr = 0.0
        evaluated_cases = 0

        for item in dataset:
            expected_ids = item.get("expected_retrieval_ids", [])
            retrieved_ids = item.get("retrieved_ids", [])
            score = self.evaluate_single(expected_ids, retrieved_ids)
            if score["available"]:
                total_hit += score["hit_rate"]
                total_mrr += score["mrr"]
                evaluated_cases += 1

        if evaluated_cases == 0:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0, "evaluated_cases": 0, "coverage": 0.0}

        return {
            "avg_hit_rate": total_hit / evaluated_cases,
            "avg_mrr": total_mrr / evaluated_cases,
            "evaluated_cases": evaluated_cases,
            "coverage": evaluated_cases / len(dataset),
        }
