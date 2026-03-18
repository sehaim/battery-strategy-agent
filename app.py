#!/usr/bin/env python3
"""
배터리 시장 전략 분석 Agent
Usage:
  python app.py
  python app.py --query "LG에너지솔루션과 CATL의 포트폴리오 전략 비교"
  python app.py --rebuild-index
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def check_api_keys():
    missing = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if missing:
        logger.warning(f"환경변수 미설정: {missing}")
        return False
    return True


def rebuild_index():
    """FAISS 인덱스 재구축"""
    from retrieval.pdf_loader import load_pdfs, chunk_pages
    from retrieval.embedder import FAISSVectorStore

    for name, data_dir, cache_path in [
        ("LG",     "data/lg",     ".cache/lg_index"),
        ("CATL",   "data/catl",   ".cache/catl_index"),
        ("Market", "data/market", ".cache/market_index"),
    ]:
        logger.info(f"[{name}] 인덱스 구축 중...")
        pages = load_pdfs(data_dir)
        if not pages:
            logger.warning(f"{data_dir}/ PDF 없음 — 건너뜀")
            continue
        chunks = chunk_pages(pages)
        store = FAISSVectorStore(chunks)
        store.save(cache_path)
        logger.info(f"[{name}] 저장 완료: {cache_path}")


def run(query: str):
    from graph import app

    initial_state = {
        "query":             query,
        "lg_rag_result":     None,
        "catl_rag_result":   None,
        "web_positive":      None,
        "web_negative":      None,
        "web_neutral":       None,
        "market_background": None,
        "merged_data":       None,
        "swot_lg":           None,
        "swot_catl":         None,
        "comparison":        None,
        "implications":      None,
        "draft_report":      None,
        "final_report":      None,
        "references":        [],
        "quality_passed":    False,
        "revision_feedback": None,
        "revision_count":    0,
        "error_log":         [],
        "retry_count":       0,
        "max_retries":       2,
        "current_step":      "start",
        "fallback_triggered": False,
        "is_complete":       False,
    }

    logger.info("=" * 60)
    logger.info(f"분석 시작: {query}")
    logger.info("=" * 60)

    result = app.invoke(initial_state)

    if result.get("final_report"):
        logger.info("=" * 60)
        logger.info("분석 완료! 보고서 위치:")
        logger.info("  outputs/battery_strategy_report.md")
        logger.info("  outputs/battery_strategy_report.pdf")
        logger.info("=" * 60)
        if result.get("error_log"):
            logger.warning(f"에러 로그: {result['error_log']}")
    else:
        logger.error("보고서 생성 실패")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="배터리 시장 전략 분석 Agent")
    parser.add_argument(
        "--query", type=str,
        default="LG에너지솔루션과 CATL의 포트폴리오 다각화 전략을 비교 분석하라",
    )
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    if not check_api_keys():
        sys.exit(1)

    if args.rebuild_index:
        rebuild_index()

    run(args.query)
