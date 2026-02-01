"""
Contract loading and PDF text extraction service.
"""

from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import pdfplumber
import structlog

from kggen_cuad.config import get_settings
from kggen_cuad.models.contract import Contract, ContractSection

logger = structlog.get_logger(__name__)


class ContractLoader:
    """
    Service for loading and processing contract documents.

    Handles PDF text extraction and contract metadata extraction.
    """

    def __init__(self):
        self.settings = get_settings()

    def load_pdf(self, file_path: Path | str) -> Contract:
        """
        Load a contract from a PDF file.

        Extracts text, page count, and creates contract sections.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Contract file not found: {file_path}")

        if not file_path.suffix.lower() == ".pdf":
            raise ValueError(f"Expected PDF file, got: {file_path.suffix}")

        # Extract text from PDF
        text, pages, sections = self._extract_pdf_text(file_path)

        # Generate CUAD ID from filename if not provided
        cuad_id = file_path.stem.replace(" ", "_").lower()

        contract = Contract(
            id=uuid4(),
            cuad_id=cuad_id,
            filename=file_path.name,
            raw_text=text,
            page_count=pages,
            word_count=len(text.split()) if text else 0,
            sections=sections,
        )

        logger.info(
            "contract_loaded",
            filename=file_path.name,
            pages=pages,
            words=contract.word_count,
        )

        return contract

    def _extract_pdf_text(
        self, file_path: Path
    ) -> tuple[str, int, list[ContractSection]]:
        """
        Extract text from PDF using pdfplumber.

        Returns (full_text, page_count, sections).
        """
        full_text_parts = []
        sections = []
        page_count = 0
        char_offset = 0

        try:
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)

                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text() or ""

                    if page_text.strip():
                        full_text_parts.append(page_text)

                        # Create section for this page
                        section = ContractSection(
                            contract_id=uuid4(),  # Will be updated
                            title=f"Page {page_num}",
                            text=page_text,
                            start_page=page_num,
                            end_page=page_num,
                            start_char=char_offset,
                            end_char=char_offset + len(page_text),
                        )
                        sections.append(section)
                        char_offset += len(page_text) + 1  # +1 for separator

            full_text = "\n\n".join(full_text_parts)
            return full_text, page_count, sections

        except Exception as e:
            logger.error("pdf_extraction_failed", file=str(file_path), error=str(e))
            raise

    def load_text(self, text: str, cuad_id: str, filename: str = "text_input") -> Contract:
        """
        Create a contract from raw text input.

        Useful for testing or non-PDF sources.
        """
        # Create simple sections by splitting on double newlines
        section_texts = text.split("\n\n")
        sections = []
        char_offset = 0

        contract_id = uuid4()

        for i, section_text in enumerate(section_texts):
            if section_text.strip():
                section = ContractSection(
                    contract_id=contract_id,
                    title=f"Section {i + 1}",
                    text=section_text,
                    start_page=1,
                    end_page=1,
                    start_char=char_offset,
                    end_char=char_offset + len(section_text),
                )
                sections.append(section)
                char_offset += len(section_text) + 2  # +2 for \n\n

        contract = Contract(
            id=contract_id,
            cuad_id=cuad_id,
            filename=filename,
            raw_text=text,
            page_count=1,
            word_count=len(text.split()),
            sections=sections,
        )

        return contract

    def chunk_contract(
        self,
        contract: Contract,
        chunk_size: int = 1000,
        overlap: int = 100,
    ) -> list[ContractSection]:
        """
        Split contract text into overlapping chunks for processing.

        Returns list of ContractSection objects with chunked text.
        """
        if not contract.raw_text:
            return []

        text = contract.raw_text
        chunks = []
        start = 0
        chunk_num = 0

        while start < len(text):
            # Find end of chunk
            end = min(start + chunk_size, len(text))

            # Try to end at a sentence boundary
            if end < len(text):
                # Look for sentence endings
                for boundary in [". ", ".\n", "? ", "?\n", "! ", "!\n"]:
                    last_boundary = text[start:end].rfind(boundary)
                    if last_boundary > chunk_size // 2:
                        end = start + last_boundary + len(boundary)
                        break

            chunk_text = text[start:end].strip()

            if chunk_text:
                # Determine page number (approximate)
                page_num = 1
                if contract.sections:
                    for section in contract.sections:
                        if section.start_char <= start < section.end_char:
                            page_num = section.start_page
                            break

                chunk = ContractSection(
                    contract_id=contract.id,
                    title=f"Chunk {chunk_num + 1}",
                    text=chunk_text,
                    start_page=page_num,
                    end_page=page_num,
                    start_char=start,
                    end_char=end,
                    section_type="chunk",
                )
                chunks.append(chunk)
                chunk_num += 1

            # Move start with overlap
            start = end - overlap
            if start >= len(text) - overlap:
                break

        logger.debug("contract_chunked", contract_id=str(contract.id), chunks=len(chunks))
        return chunks

    def identify_contract_type(self, contract: Contract) -> str | None:
        """
        Attempt to identify the contract type from text content.

        Uses keyword matching for common contract types.
        """
        if not contract.raw_text:
            return None

        text_lower = contract.raw_text.lower()[:5000]  # Check first 5000 chars

        type_keywords = {
            "License Agreement": [
                "license agreement",
                "software license",
                "licensing agreement",
                "grant of license",
            ],
            "Service Agreement": [
                "service agreement",
                "services agreement",
                "master service",
                "professional services",
            ],
            "Development Agreement": [
                "development agreement",
                "software development",
                "development services",
            ],
            "Non-Disclosure Agreement": [
                "non-disclosure",
                "nda",
                "confidentiality agreement",
                "confidential information",
            ],
            "Employment Agreement": [
                "employment agreement",
                "employment contract",
                "employee agreement",
            ],
            "Partnership Agreement": [
                "partnership agreement",
                "joint venture",
            ],
            "Purchase Agreement": [
                "purchase agreement",
                "asset purchase",
                "acquisition agreement",
            ],
            "Lease Agreement": [
                "lease agreement",
                "rental agreement",
            ],
        }

        for contract_type, keywords in type_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    contract.contract_type = contract_type
                    return contract_type

        return None

    def extract_parties(self, contract: Contract) -> list[str]:
        """
        Extract party names from contract text.

        Uses common patterns like "between X and Y".
        """
        if not contract.raw_text:
            return []

        import re

        parties = []
        text = contract.raw_text[:3000]  # Check beginning

        # Pattern: "between [PARTY A] and [PARTY B]"
        between_pattern = r'between\s+([^,]+?)\s+(?:and|&)\s+([^,\n]+)'
        matches = re.findall(between_pattern, text, re.IGNORECASE)
        for match in matches:
            parties.extend([m.strip() for m in match if m.strip()])

        # Pattern: "hereinafter referred to as" or "hereinafter called"
        hereinafter_pattern = r'([A-Z][A-Za-z\s,\.]+?)\s*(?:\(|,)?\s*hereinafter'
        matches = re.findall(hereinafter_pattern, text)
        parties.extend([m.strip() for m in matches if m.strip()])

        # Deduplicate and clean
        cleaned_parties = []
        seen = set()
        for party in parties:
            # Remove common prefixes/suffixes
            party = party.strip(' ,.()"\'')
            if len(party) > 2 and party.lower() not in seen:
                seen.add(party.lower())
                cleaned_parties.append(party)

        contract.parties = cleaned_parties[:10]  # Limit to 10 parties
        return cleaned_parties


@lru_cache()
def get_contract_loader() -> ContractLoader:
    """Get cached contract loader instance."""
    return ContractLoader()
