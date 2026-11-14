"""Read and normalize source data from Vinamilk 2025 Annual Report."""
import re
import unicodedata
from pathlib import Path
import pypdf

from src.models import DocumentPage


def determine_section(page_num: int, header_text: str) -> str:
    """Map physical page and running header text to canonical section ID."""
    lower_header = header_text.lower()
    
    if "báo cáo tài chính" in lower_header or page_num >= 106:
        return "bao_cao_tai_chinh"
    if "phát triển bền vững" in lower_header or (101 <= page_num <= 105):
        return "phat_trien_ben_vung"
    if "báo cáo quản trị" in lower_header or "quản trị rủi ro" in lower_header or (66 <= page_num <= 100):
        return "quan_tri_cong_ty"
    if any(k in lower_header for k in ["báo cáo của hđqt", "báo cáo của ban điều hành", "tổng quan ngành sữa"]) or (26 <= page_num <= 65):
        return "bao_cao_hoat_dong"
    if any(k in lower_header for k in ["thông tin chung", "nhân sự chủ chốt", "lịch sử hình thành"]) or (6 <= page_num <= 25):
        return "thong_tin_chung"
    return "mo_dau"


def normalize_text(text: str) -> str:
    """Normalize Unicode (NFC), remove non-printable characters, standardise whitespaces."""
    if not text:
        return ""
    # Unicode Normalization Form C
    text = unicodedata.normalize("NFC", text)
    # Replace non-breaking spaces and other special spaces
    text = text.replace("\xa0", " ").replace("\u200b", "")
    # Normalize multiple whitespaces within a line
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize excessive blank lines (more than 2 consecutive newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(
    path: Path | str,
    document_id: str = "VNM_AR_2025",
    source: str = "20260319_VNM_Bao_cao_thuong_nien_2025_eae619d5f6.pdf",
    min_char_count: int = 150
) -> tuple[list[DocumentPage], list[dict]]:
    """
    Extract text per physical page, filter out cover / empty pages.
    Returns:
        (useful_pages, excluded_pages_log)
    """
    reader = pypdf.PdfReader(str(path))
    useful_pages: list[DocumentPage] = []
    excluded_pages_log: list[dict] = []

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        raw_text = page.extract_text() or ""
        clean_text = normalize_text(raw_text)
        char_count = len(clean_text)

        # First line usually holds running header
        first_line = clean_text.split("\n")[0] if clean_text else ""
        section = determine_section(page_num, first_line)

        if char_count < min_char_count:
            excluded_pages_log.append({
                "page": page_num,
                "reason": "Character count below threshold (cover/blank/divider)",
                "char_count": char_count
            })
            continue

        doc_page = DocumentPage(
            document_id=document_id,
            source=source,
            page=page_num,
            text=clean_text,
            char_count=char_count,
            metadata={
                "document_id": document_id,
                "source": source,
                "page": page_num,
                "year": 2025,
                "company": "Vinamilk",
                "section": section
            }
        )
        useful_pages.append(doc_page)

    return useful_pages, excluded_pages_log
