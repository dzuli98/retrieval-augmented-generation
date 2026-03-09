from typing import Dict, List, Tuple

from ..config import Config
from ..logger import get_logger
from ..models import Claim, ConflictInfo


class MediatorAgent:

    def __init__(self, config: Config):
        self.config = config

    def detect_conflicts(self, claims: List[Claim]) -> ConflictInfo:
        if len(claims) < 2:
            return ConflictInfo(
                has_conflict=False,
                conflicting_answers=[],
                explanation="Not enough claims to compare",
            )

        answers = [c.answer.lower().strip() for c in claims]
        unique_answers = list(set(answers))

        if len(unique_answers) == 1:
            return ConflictInfo(
                has_conflict=False,
                conflicting_answers=[],
                explanation="All claims agree",
            )

        return ConflictInfo(
            has_conflict=True,
            conflicting_answers=unique_answers,
            explanation=f"Found {len(unique_answers)} different answers: {unique_answers}",
        )

    def resolve_by_majority_vote(
        self, claims: List[Claim]
    ) -> Tuple[str, List[Claim], List[Claim]]:

        if not claims:
            return "", [], []

        answer_scores: Dict[str, float] = {}
        answer_claims: Dict[str, List[Claim]] = {}

        for claim in claims:
            normalized = claim.answer.lower().strip()
            weighted_score = claim.confidence * claim.reliability_score
            answer_scores[normalized] = (
                answer_scores.get(normalized, 0) + weighted_score
            )
            if normalized not in answer_claims:
                answer_claims[normalized] = []
            answer_claims[normalized].append(claim)

        winner = max(answer_scores.keys(), key=lambda x: answer_scores[x])
        supporting = answer_claims[winner]
        rejected = [c for ans, cs in answer_claims.items() if ans != winner for c in cs]
        original_answer = supporting[0].answer if supporting else winner

        return original_answer, supporting, rejected

    def reconcile(
        self, query: str, claims: List[Claim]
    ) -> Tuple[str, List[Claim], List[Claim], str]:
        conflict_info = self.detect_conflicts(claims)
        if not conflict_info.has_conflict:
            return (
                claims[0].answer if claims else "",
                claims,
                [],
                "All sources agree on this answer.",
            )

        logger = get_logger()
        logger.warning(f"Conflict detected: {conflict_info.explanation}")

        answer, supporting, rejected = self.resolve_by_majority_vote(claims)
        explanation = (
            f"Resolved by reliability-weighted majority vote. "
            f"{len(supporting)} sources support '{answer}', {len(rejected)} rejected."
        )
        return answer, supporting, rejected, explanation
