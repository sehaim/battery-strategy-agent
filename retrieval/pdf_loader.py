import os
import logging
from pathlib import Path
from typing import List, Tuple
from pypdf import PdfReader

logger = logging.getLogger(__name__)

MAX_PAGES   = int(os.getenv("MAX_PDF_PAGES", "100"))
CHUNK_SIZE  = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))


def load_pdfs(directory: str) -> List[Tuple[str, str, int]]:
    """디렉토리 내 PDF를 로드. 반환: [(텍스트, 파일명, 페이지번호)]"""
    pages, total = [], 0
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.warning(f"디렉토리 없음: {directory}")
        return pages

    for pdf_path in sorted(dir_path.glob("*.pdf")):
        if total >= MAX_PAGES:
            break
        try:
            reader = PdfReader(str(pdf_path))
            for page_num, page in enumerate(reader.pages, 1):
                if total >= MAX_PAGES:
                    break
                text = (page.extract_text() or "").strip()
                if len(text) >= 50:
                    pages.append((text, pdf_path.name, page_num))
                    total += 1
            logger.info(f"로드: {pdf_path.name}")
        except Exception as e:
            logger.error(f"PDF 로드 실패 [{pdf_path.name}]: {e}")

    logger.info(f"총 {total}p 로드 ({directory})")
    return pages


def chunk_pages(pages: List[Tuple[str, str, int]]) -> List[dict]:
    """페이지를 청크로 분할"""
    chunks, idx = [], 0
    for text, filename, page_num in pages:
        start = 0
        while start < len(text):
            chunk = text[start:start + CHUNK_SIZE]
            if len(chunk.strip()) > 30:
                chunks.append({
                    "text": chunk,
                    "source": filename,
                    "page": page_num,
                    "chunk_id": idx,
                })
                idx += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP
    logger.info(f"총 {len(chunks)}개 청크 생성")
    return chunks
