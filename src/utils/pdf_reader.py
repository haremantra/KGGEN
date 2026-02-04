"""PDF contract reader utility."""

from pathlib import Path
import pdfplumber


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text content.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    return "\n\n".join(text_parts)


def extract_text_with_metadata(pdf_path: str | Path) -> dict:
    """Extract text and metadata from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict with 'text', 'metadata', and 'pages' keys.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        metadata = pdf.metadata or {}

        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            pages.append({
                "page_number": i + 1,
                "text": page_text,
                "width": page.width,
                "height": page.height,
            })

    full_text = "\n\n".join(p["text"] for p in pages if p["text"])

    return {
        "text": full_text,
        "metadata": metadata,
        "pages": pages,
        "total_pages": len(pages),
        "file_path": str(pdf_path),
    }
