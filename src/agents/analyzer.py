import asyncio
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

from openai import OpenAI

from ..config import Config
from ..logger import get_logger
from ..models import Claim, Document

ANALYZER_PROMPT = """You are an expert document analyzer. Given a question and a document, extract the answer or claim the document provides.

Respond in JSON format:
{
    "has_answer": true/false,
    "answer": "the extracted answer",
    "confidence": 0.0-1.0,
    "supporting_quote": "exact quote from document"
}

Be conservative with confidence:
- 0.9-1.0: Document explicitly answers the question
- 0.7-0.9: Strong implication
- 0.5-0.7: Partial answer
- Below 0.5: Weak or no answer"""


@dataclass
class AnalysisResult:
    doc_id: str
    claim: Optional[Claim]
    success: bool
    error: Optional[str] = None


class DocumentAnalyzer:

    def __init__(self, config: Config, instance_id: int = 0):
        self.config = config
        self.instance_id = instance_id
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = OpenAI(api_key=self.config.openai_api_key)
        return self._client

    async def analyze(self, query: str, doc: Document) -> AnalysisResult:
        try:
            client = self._get_client()

            user_prompt = f"""Question: {query}

Document ID: {doc.doc_id}
Document Content:
{doc.content[:2000]}

Extract the answer/claim from this document."""

            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": ANALYZER_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)

            claim = None
            if result.get("has_answer"):
                claim = Claim(
                    doc_id=doc.doc_id,
                    answer=result.get("answer", ""),
                    confidence=float(result.get("confidence", 0)),
                    supporting_quote=result.get("supporting_quote", ""),
                    source_label=doc.label,
                )

            return AnalysisResult(doc_id=doc.doc_id, claim=claim, success=True)

        except Exception as e:
            return AnalysisResult(
                doc_id=doc.doc_id, claim=None, success=False, error=str(e)
            )


class ParallelAnalyzer:

    def __init__(self, config: Config):
        self.config = config

    async def analyze_documents(
        self, query: str, documents: List[Document], max_parallel: int = 5
    ) -> Tuple[List[Claim], List[AnalysisResult]]:

        if not documents:
            return [], []

        logger = get_logger()
        logger.info(
            f"Analyzing {len(documents)} documents in parallel (max {max_parallel} concurrent)..."
        )

        tasks = []
        for i, doc in enumerate(documents):
            analyzer = DocumentAnalyzer(self.config, instance_id=i)
            tasks.append(analyzer.analyze(query, doc))

        semaphore = asyncio.Semaphore(max_parallel)

        async def bounded_task(task):
            async with semaphore:
                return await task

        bounded_tasks = [bounded_task(task) for task in tasks]
        results = await asyncio.gather(*bounded_tasks, return_exceptions=True)

        final_results: List[AnalysisResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(
                    AnalysisResult(
                        doc_id=documents[i].doc_id,
                        claim=None,
                        success=False,
                        error=str(result),
                    )
                )
            else:
                final_results.append(result)
                if result.success and result.claim:
                    logger.debug(f"{result.doc_id}: '{result.claim.answer[:50]}...'")
                elif not result.success:
                    logger.warning(f"{result.doc_id}: {result.error}")

        claims = [r.claim for r in final_results if r.success and r.claim]
        logger.info(f"Extracted {len(claims)} claims from {len(documents)} documents")

        return claims, final_results
