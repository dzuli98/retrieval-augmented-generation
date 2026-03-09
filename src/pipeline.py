import asyncio
import logging
from datetime import datetime
from typing import List

from .agents import MediatorAgent, ParallelAnalyzer, RetrieverAgent
from .config import Config
from .data_loader import RAMDocsLoader
from .logger import get_logger
from .models import Document, FinalAnswer


class RAGPipeline:

    def __init__(self, config: Config):
        self.config = config
        self.retriever = RetrieverAgent(config)
        self.analyzer = ParallelAnalyzer(config)
        self.mediator = MediatorAgent(config)
        self.trace: List[str] = []
        self.logger = get_logger()
        self.questions: List[str] = []

    def _log(self, message: str, level: int = logging.INFO):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.trace.append(entry)
        self.logger.log(level, message)

    def load_and_index(
        self, num_samples: int = 10, documents: List[Document] = None
    ) -> None:
        if documents:
            self._log(f"Using {len(documents)} provided documents")
            all_docs = documents
        else:
            self._log("Loading RAMDocs dataset from HuggingFace...")
            loader = RAMDocsLoader()
            loader.load(num_samples=num_samples)
            all_docs = loader.get_all_documents()
            self.questions = [
                item.get("question", "")
                for item in loader.questions_data
                if item.get("question")
            ]
            self._log(f"Loaded {len(all_docs)} documents from {num_samples} questions")

        self._log("Building FAISS index...")
        self.retriever.build_index(all_docs)
        self._log("FAISS index ready!")

    def query(self, question: str) -> FinalAnswer:

        self.trace = []
        print("\n" + "=" * 100)
        self._log(f"Step 1: Retrieving documents for: '{question}'")
        retrieved_docs = self.retriever.retrieve(question, top_k=self.config.top_k)
        self._log(f"   Retrieved {len(retrieved_docs)} documents")

        for doc in retrieved_docs:
            self._log(
                f"   - {doc.doc_id} (score={doc.relevance_score:.3f}, label={doc.label})"
            )

        if not retrieved_docs:
            return FinalAnswer(
                query=question,
                answer="No relevant documents found.",
                confidence=0.0,
                supporting_docs=[],
                rejected_docs=[],
                reconciliation_explanation="No documents retrieved.",
                trace=self.trace,
            )

        self._log(f"Step 2: Analyzing documents in parallel...")
        claims, analysis_results = asyncio.run(
            self.analyzer.analyze_documents(question, retrieved_docs)
        )
        self._log(
            f"   Extracted {len(claims)} claims from {len(analysis_results)} documents:"
        )
        for claim in claims:
            self._log(
                f"   - {claim.doc_id}: '{claim.answer}' (conf={claim.confidence:.0%}, reliability={claim.reliability_score:.0%})"
            )

        if not claims:
            return FinalAnswer(
                query=question,
                answer="Could not extract answers from retrieved documents.",
                confidence=0.0,
                supporting_docs=[],
                rejected_docs=[],
                reconciliation_explanation="No claims extracted.",
                trace=self.trace,
            )

        self._log(f"Step 3: Detecting conflicts and reconciling...")
        conflict_info = self.mediator.detect_conflicts(claims)
        self._log(f"   Conflict detected: {conflict_info.has_conflict}")
        final_answer, supporting, rejected, explanation = self.mediator.reconcile(
            question, claims
        )

        self._log(f"   Resolution: '{final_answer}' via majority vote + reliability")
        self._log(f"   Supporting: {len(supporting)}, Rejected: {len(rejected)}")

        self._log(f"Step 4: Generating final answer...")
        if supporting:
            avg_confidence = sum(c.confidence for c in supporting) / len(supporting)
        else:
            avg_confidence = 0.0

        supporting_docs = [
            {
                "doc_id": c.doc_id,
                "answer": c.answer,
                "confidence": c.confidence,
                "reliability": c.reliability_score,
                "source_label": c.source_label,
                "quote": c.supporting_quote,
            }
            for c in supporting
        ]

        rejected_docs = [
            {
                "doc_id": c.doc_id,
                "answer": c.answer,
                "source_label": c.source_label,
                "reason": f"Conflict with majority (source={c.source_label})",
            }
            for c in rejected
        ]

        self._log(
            f"✅ Final answer: '{final_answer}' (confidence={avg_confidence:.0%})"
        )
        return FinalAnswer(
            query=question,
            answer=final_answer,
            confidence=avg_confidence,
            supporting_docs=supporting_docs,
            rejected_docs=rejected_docs,
            reconciliation_explanation=explanation,
            trace=self.trace,
        )
