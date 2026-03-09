from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    doc_id: str
    content: str
    label: str = "unknown"
    relevance_score: float = 0.0


@dataclass
class Claim:
    doc_id: str
    answer: str
    confidence: float
    supporting_quote: str
    source_label: str = "unknown"

    @property
    def reliability_score(self) -> float:
        label_weights = {
            "correct": 1.0,
            "unknown": 0.7,
            "noise": 0.3,
            "misinfo": 0.1,
        }
        return label_weights.get(self.source_label, 0.7)


@dataclass
class ConflictInfo:
    has_conflict: bool
    conflicting_answers: List[str]
    explanation: str


@dataclass
class FinalAnswer:
    query: str
    answer: str
    confidence: float
    supporting_docs: List[Dict[str, Any]]
    rejected_docs: List[Dict[str, Any]]
    reconciliation_explanation: str
    trace: List[str]  # Log of steps taken

    def to_readable(self) -> str:
        lines = [
            "=" * 60,
            f"QUERY: {self.query}",
            "=" * 60,
            f"\n📝 ANSWER: {self.answer}",
            f"   Confidence: {self.confidence:.0%}",
            "\n📚 SUPPORTING EVIDENCE:",
        ]
        for doc in self.supporting_docs:
            reliability = doc.get("reliability", 1.0)
            label = doc.get("source_label", "unknown")
            lines.append(
                f"   • [{doc['doc_id']}] confidence={doc['confidence']:.0%}, reliability={reliability:.0%} ({label})"
            )
            quote = doc.get("quote", "")
            lines.append(
                f'     "{quote[:80]}..."' if len(quote) > 80 else f'     "{quote}"'
            )
        if self.rejected_docs:
            lines.append("\n❌ REJECTED CLAIMS:")
            for doc in self.rejected_docs:
                lines.append(
                    f"   • [{doc['doc_id']}] \"{doc['answer']}\" - {doc['reason']}"
                )

        lines.append(f"\n🔍 RECONCILIATION:\n   {self.reconciliation_explanation}")
        lines.append("\n📋 EXECUTION TRACE:")
        for step in self.trace:
            lines.append(f"   → {step}")
        return "\n".join(lines)
