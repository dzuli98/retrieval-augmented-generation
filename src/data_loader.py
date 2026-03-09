from typing import Dict, List

from datasets import load_dataset

from .logger import get_logger
from .models import Document


class RAMDocsLoader:

    DATASET_NAME = "HanNight/RAMDocs"

    def __init__(self):
        self.dataset = None
        self.questions_data = []

    def load(self, num_samples: int = 10, split: str = "test") -> List[Dict]:
        logger = get_logger()
        logger.info(
            f"Loading RAMDocs dataset from HuggingFace ({num_samples} samples)..."
        )

        self.dataset = load_dataset(self.DATASET_NAME, split=split)

        self.questions_data = []
        for item in self.dataset:
            if len(self.questions_data) >= num_samples:
                break
            if item.get("documents"):
                self.questions_data.append(item)

        logger.info(f"Loaded {len(self.questions_data)} question-document groups")
        return self.questions_data

    def get_all_documents(self) -> List[Document]:
        documents = []

        for item in self.questions_data:
            for i, doc_data in enumerate(item.get("documents", [])):
                doc = Document(
                    doc_id=f"q{self.questions_data.index(item)}_d{i}",
                    content=doc_data.get("text", ""),
                    label=doc_data.get("type", "unknown"),  # RAMDocs uses 'type' field
                )
                documents.append(doc)

        return documents

    def get_ground_truth(self) -> Dict[str, Dict]:
        ground_truth = {}
        for item in self.questions_data:
            question = item.get("question", "")

            gold_answers = item.get("gold_answers", [])
            wrong_answers = item.get("wrong_answers", [])

            for doc in item.get("documents", []):
                if doc.get("type") == "correct" and doc.get("answer"):
                    if doc["answer"] not in gold_answers:
                        gold_answers.append(doc["answer"])

            if gold_answers:
                ground_truth[question] = {
                    "gold_answers": gold_answers,
                    "wrong_answers": wrong_answers,
                }

        return ground_truth
