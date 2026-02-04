"""Contract Clause Classifier using CUAD labels with semantic similarity."""

import re
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from .cuad_labels import CUAD_LABELS, get_all_labels
from ..config import settings


@dataclass
class ClauseLabel:
    """A label prediction for a clause."""
    label: str
    confidence: float
    category: str
    description: str


@dataclass
class ClassifiedClause:
    """A clause with its predicted labels."""
    text: str
    clause_index: int
    labels: list[ClauseLabel]


class ClauseClassifier:
    """Classifies contract clauses using CUAD label categories with semantic similarity."""

    COLLECTION_NAME = "cuad_labels"

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_qdrant: bool = True,
    ):
        """Initialize the classifier.

        Args:
            model_name: Sentence transformer model for embeddings.
            use_qdrant: Whether to use Qdrant for vector storage (vs in-memory).
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.use_qdrant = use_qdrant
        self._label_embeddings = None
        self._labels = None

        if use_qdrant:
            self.qdrant = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
            )
        else:
            self.qdrant = None

    def initialize(self):
        """Initialize the classifier by building label embeddings."""
        self._labels = get_all_labels()

        # Build rich text representations for each label
        label_texts = []
        for label in self._labels:
            info = CUAD_LABELS[label]
            # Combine label name, description, and patterns for better matching
            patterns = " ".join(info.get("patterns", []))
            text = f"{label}: {info['description']}. Examples: {patterns}"
            label_texts.append(text)

        # Generate embeddings
        print(f"Generating embeddings for {len(self._labels)} CUAD labels...")
        self._label_embeddings = self.model.encode(
            label_texts,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        if self.use_qdrant:
            self._store_in_qdrant()

        print("Classifier initialized.")

    def _store_in_qdrant(self):
        """Store label embeddings in Qdrant for efficient similarity search."""
        vector_size = self._label_embeddings.shape[1]

        # Recreate collection
        try:
            self.qdrant.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass

        self.qdrant.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

        # Insert label embeddings
        points = [
            PointStruct(
                id=i,
                vector=self._label_embeddings[i].tolist(),
                payload={
                    "label": self._labels[i],
                    "category": CUAD_LABELS[self._labels[i]]["category"],
                    "description": CUAD_LABELS[self._labels[i]]["description"],
                },
            )
            for i in range(len(self._labels))
        ]

        self.qdrant.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points,
        )

    def classify_clause(
        self,
        clause_text: str,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> list[ClauseLabel]:
        """Classify a single clause and return top matching labels.

        Args:
            clause_text: The clause text to classify.
            top_k: Number of top labels to return.
            threshold: Minimum similarity score to include.

        Returns:
            List of ClauseLabel with confidence scores.
        """
        if self._label_embeddings is None:
            self.initialize()

        # Embed the clause
        clause_embedding = self.model.encode(clause_text, convert_to_numpy=True)

        if self.use_qdrant:
            # Search in Qdrant (using query_points for newer client versions)
            try:
                results = self.qdrant.query_points(
                    collection_name=self.COLLECTION_NAME,
                    query=clause_embedding.tolist(),
                    limit=top_k,
                ).points
            except AttributeError:
                # Fall back to older API
                results = self.qdrant.search(
                    collection_name=self.COLLECTION_NAME,
                    query_vector=clause_embedding.tolist(),
                    limit=top_k,
                )

            labels = []
            for result in results:
                score = getattr(result, 'score', 0.0)
                if score >= threshold:
                    payload = result.payload
                    labels.append(ClauseLabel(
                        label=payload["label"],
                        confidence=float(score),
                        category=payload["category"],
                        description=payload["description"],
                    ))
            return labels
        else:
            # In-memory cosine similarity
            similarities = np.dot(self._label_embeddings, clause_embedding) / (
                np.linalg.norm(self._label_embeddings, axis=1) * np.linalg.norm(clause_embedding)
            )

            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]

            labels = []
            for idx in top_indices:
                if similarities[idx] >= threshold:
                    label_name = self._labels[idx]
                    labels.append(ClauseLabel(
                        label=label_name,
                        confidence=float(similarities[idx]),
                        category=CUAD_LABELS[label_name]["category"],
                        description=CUAD_LABELS[label_name]["description"],
                    ))
            return labels

    def classify_contract(
        self,
        contract_text: str,
        top_k: int = 3,
        threshold: float = 0.35,
    ) -> list[ClassifiedClause]:
        """Classify all clauses in a contract.

        Args:
            contract_text: Full contract text.
            top_k: Number of top labels per clause.
            threshold: Minimum similarity threshold.

        Returns:
            List of ClassifiedClause objects.
        """
        # Split into clauses/paragraphs
        clauses = self._split_into_clauses(contract_text)

        results = []
        for i, clause in enumerate(clauses):
            if len(clause.strip()) < 50:  # Skip very short segments
                continue

            labels = self.classify_clause(clause, top_k=top_k, threshold=threshold)

            if labels:  # Only include if we found matching labels
                results.append(ClassifiedClause(
                    text=clause[:500] + "..." if len(clause) > 500 else clause,
                    clause_index=i,
                    labels=labels,
                ))

        return results

    def _split_into_clauses(self, text: str) -> list[str]:
        """Split contract text into individual clauses/paragraphs."""
        # Split on section numbers, paragraph breaks, or common clause markers
        # Pattern matches: "1.", "1.1", "Section 1", "(a)", etc.
        section_pattern = r'(?:\n\s*(?:\d+\.[\d.]*|\(\w\)|Section\s+\d+|ARTICLE\s+[IVX\d]+)[.\s])'

        # First try splitting by sections
        parts = re.split(section_pattern, text, flags=re.IGNORECASE)

        # If no sections found, split by double newlines
        if len(parts) <= 1:
            parts = re.split(r'\n\s*\n', text)

        # Clean up and filter
        clauses = []
        for part in parts:
            part = part.strip()
            if part and len(part) > 30:  # Minimum length
                clauses.append(part)

        return clauses

    def get_contract_summary(
        self,
        contract_text: str,
        threshold: float = 0.4,
    ) -> dict:
        """Get a summary of which CUAD categories are present in the contract.

        Args:
            contract_text: Full contract text.
            threshold: Minimum confidence threshold.

        Returns:
            Dict with category counts and top findings.
        """
        classified = self.classify_contract(contract_text, top_k=3, threshold=threshold)

        # Aggregate by label
        label_occurrences = {}
        for clause in classified:
            for label_info in clause.labels:
                if label_info.label not in label_occurrences:
                    label_occurrences[label_info.label] = {
                        "count": 0,
                        "max_confidence": 0.0,
                        "category": label_info.category,
                        "best_match": None,
                    }
                label_occurrences[label_info.label]["count"] += 1
                if label_info.confidence > label_occurrences[label_info.label]["max_confidence"]:
                    label_occurrences[label_info.label]["max_confidence"] = label_info.confidence
                    label_occurrences[label_info.label]["best_match"] = clause.text

        # Sort by confidence
        sorted_labels = sorted(
            label_occurrences.items(),
            key=lambda x: x[1]["max_confidence"],
            reverse=True,
        )

        # Group by category
        by_category = {}
        for label, info in sorted_labels:
            cat = info["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append({
                "label": label,
                "confidence": info["max_confidence"],
                "occurrences": info["count"],
            })

        return {
            "total_clauses_analyzed": len(classified),
            "labels_found": len(label_occurrences),
            "by_category": by_category,
            "top_findings": sorted_labels[:10],
        }
