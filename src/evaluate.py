import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from .config import Config
from .data_loader import RAMDocsLoader
from .logger import get_logger
from .pipeline import RAGPipeline


@dataclass
class EvaluationResult:
    question: str
    gold_answers: List[str]
    wrong_answers: List[str]
    predicted: str
    confidence: float
    is_correct: bool = False
    is_misinformation: bool = False
    exact_match: float = 0.0
    f1_score: float = 0.0
    best_gold_match: str = ""


def normalize_answer(text: str) -> str:
    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s):
        return " ".join(s.split())

    def remove_punctuation(s):
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    def lower(s):
        return s.lower()

    return white_space_fix(remove_articles(remove_punctuation(lower(text))))


def get_tokens(text: str) -> List[str]:
    return normalize_answer(text).split()


def compute_exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def compute_f1(prediction: str, gold: str) -> float:
    pred_tokens = get_tokens(prediction)
    gold_tokens = get_tokens(gold)

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def matches_any(
    prediction: str, answers: List[str], threshold: float = 0.5
) -> Tuple[bool, str]:
    if not answers:
        return False, ""

    best_f1 = 0.0
    best_match = ""
    pred_normalized = normalize_answer(prediction)

    for answer in answers:
        if compute_exact_match(prediction, answer) == 1.0:
            return True, answer

        ans_normalized = normalize_answer(answer)
        if ans_normalized in pred_normalized or pred_normalized in ans_normalized:
            return True, answer

        f1 = compute_f1(prediction, answer)
        if f1 > best_f1:
            best_f1 = f1
            best_match = answer

    return best_f1 >= threshold, best_match


def evaluate_single(
    prediction: str, gold_answers: List[str], wrong_answers: List[str]
) -> Tuple[bool, bool, float, float, str]:
    is_correct, best_gold = matches_any(prediction, gold_answers)
    is_misinfo, _ = matches_any(prediction, wrong_answers)

    best_em = 0.0
    best_f1 = 0.0

    for gold in gold_answers:
        em = compute_exact_match(prediction, gold)
        f1 = compute_f1(prediction, gold)

        if em > best_em:
            best_em = em
        if f1 > best_f1:
            best_f1 = f1
            best_gold = gold

    return is_correct, is_misinfo, best_em, best_f1, best_gold


def evaluate_pipeline(
    pipeline: RAGPipeline, questions: List[str], ground_truth: Dict[str, Dict]
) -> Tuple[List[EvaluationResult], Dict[str, float]]:
    logger = get_logger()
    results: List[EvaluationResult] = []

    for question in questions:
        gt = ground_truth.get(question)
        if not gt or not gt.get("gold_answers"):
            logger.warning(f"No ground truth for: {question}")
            continue

        gold_answers = gt["gold_answers"]
        wrong_answers = gt.get("wrong_answers", [])

        try:
            answer = pipeline.query(question)
            is_correct, is_misinfo, em, f1, best_gold = evaluate_single(
                answer.answer, gold_answers, wrong_answers
            )

            result = EvaluationResult(
                question=question,
                gold_answers=gold_answers,
                wrong_answers=wrong_answers,
                predicted=answer.answer,
                confidence=answer.confidence,
                is_correct=is_correct,
                is_misinformation=is_misinfo,
                exact_match=em,
                f1_score=f1,
                best_gold_match=best_gold,
            )
            results.append(result)

            if is_correct:
                status = "✅"
            elif is_misinfo:
                status = "⚠️ MISINFO"
            else:
                status = "❌"

            logger.info(f"{status} Q: {question[:50]}...")
            logger.info(f"   Gold: {gold_answers[0] if gold_answers else 'N/A'}")
            logger.info(
                f"   Pred: {answer.answer} (F1={f1:.2f}, conf={answer.confidence:.0%})"
            )

        except Exception as e:
            logger.error(f"Error processing '{question}': {e}")

    total = len(results)
    if total == 0:
        return results, {"total": 0, "accuracy": 0, "f1": 0, "em": 0, "misinfo_rate": 0}

    correct = sum(1 for r in results if r.is_correct)
    misinfo = sum(1 for r in results if r.is_misinformation)

    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "exact_match": sum(r.exact_match for r in results) / total,
        "f1": sum(r.f1_score for r in results) / total,
        "misinfo_count": misinfo,
        "misinfo_rate": misinfo / total,
        "avg_confidence": sum(r.confidence for r in results) / total,
    }

    return results, metrics


def run_evaluation(api_key: str, num_samples: int = 5) -> Dict[str, float]:

    logger = get_logger()
    logger.info(f"Starting evaluation with {num_samples} samples")

    config = Config.from_env(api_key)
    pipeline = RAGPipeline(config)
    loader = RAMDocsLoader()
    loader.load(num_samples=num_samples)
    ground_truth = loader.get_ground_truth()
    questions = list(ground_truth.keys())
    logger.info(f"Loaded {len(questions)} questions with ground truth")

    pipeline.load_and_index(num_samples=num_samples)
    results, metrics = evaluate_pipeline(pipeline, questions, ground_truth)

    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total questions:     {metrics['total']}")
    logger.info(f"Correct:             {metrics['correct']}")
    logger.info(f"Accuracy:            {metrics['accuracy']:.1%}")
    logger.info(f"Exact Match:         {metrics['exact_match']:.1%}")
    logger.info(f"F1 Score:            {metrics['f1']:.2f}")
    logger.info(
        f"Misinformation:      {metrics['misinfo_count']} ({metrics['misinfo_rate']:.1%})"
    )
    logger.info(f"Avg Confidence:      {metrics['avg_confidence']:.1%}")
    logger.info("=" * 60)

    return metrics


if __name__ == "__main__":
    import sys

    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    if not api_key:
        print("Usage: python -m src.evaluate <api_key>")
        sys.exit(1)

    run_evaluation(api_key, num_samples=5)
