from __future__ import annotations
import os
import logging
import threading
from collections import defaultdict
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from agents.rag_base import build_store, run_rag
from prompts.prompts import WEB_QUERIES, WEB_SYNTHESIS_PROMPT, MARKET_RAG_PROMPT

logger = logging.getLogger(__name__)

MAX_PER_DOMAIN = 2

MARKET_DATA_DIR  = "data/market"
MARKET_CACHE_PATH = ".cache/market_index"

_market_store = None
_market_lock  = threading.Lock()


def _get_market_store():
    global _market_store
    with _market_lock:
        if _market_store is None:
            _market_store = build_store(MARKET_DATA_DIR, MARKET_CACHE_PATH)
    return _market_store


def _get_search_tool():
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        logger.warning("TAVILY_API_KEY 미설정 → DuckDuckGo fallback 사용 (검색 품질 저하 가능)")
        from langchain_community.tools import DuckDuckGoSearchRun
        return DuckDuckGoSearchRun()
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        return TavilySearchResults(max_results=5)
    except Exception as e:
        logger.warning(f"Tavily 초기화 실패 ({e}) → DuckDuckGo fallback 사용")
        from langchain_community.tools import DuckDuckGoSearchRun
        return DuckDuckGoSearchRun()


def _domain(url: str) -> str:
    try:
        return url.split("/")[2]
    except Exception:
        return url


def _deduplicate_by_domain(results: list) -> list:
    """동일 도메인 최대 MAX_PER_DOMAIN건만 허용"""
    counter = defaultdict(int)
    filtered = []
    for r in results:
        url = r.get("url", "") if isinstance(r, dict) else str(r)
        d = _domain(url)
        if counter[d] < MAX_PER_DOMAIN:
            filtered.append(r)
            counter[d] += 1
    return filtered


def _search(tool, queries: List[str]) -> tuple[str, list[str]]:
    """여러 쿼리 실행 후 중복 제거. (텍스트, 출처URL목록) 튜플 반환"""
    all_results = []
    for q in queries:
        try:
            res = tool.invoke(q)
            if isinstance(res, list):
                all_results.extend(res)
            else:
                all_results.append({"content": str(res), "url": ""})
        except Exception as e:
            logger.warning(f"검색 실패 [{q}]: {e}")

    filtered = _deduplicate_by_domain(all_results)

    texts   = []
    sources = []
    for r in filtered:
        if isinstance(r, dict):
            content = r.get("content", r.get("snippet", str(r)))
            url = r.get("url", "")
            texts.append(f"{content}\n출처: {url}" if url else content)
            if url:
                sources.append(url)
        else:
            texts.append(str(r))
    return "\n\n".join(texts) or "검색 결과 없음", sources


def web_search_node(state: AgentState) -> dict:
    logger.info("[Web Search] 시작 — 이중 쿼리 편향 방지")
    try:
        tool = _get_search_tool()
        llm  = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0.1)

        positive_raw1, src_pos1 = _search(tool, WEB_QUERIES["lg_positive"])
        positive_raw2, src_pos2 = _search(tool, WEB_QUERIES["catl_positive"])
        negative_raw1, src_neg1 = _search(tool, WEB_QUERIES["lg_negative"])
        negative_raw2, src_neg2 = _search(tool, WEB_QUERIES["catl_negative"])
        neutral_raw,   src_neu  = _search(tool, WEB_QUERIES["market"])

        positive_raw = positive_raw1 + "\n\n" + positive_raw2
        negative_raw = negative_raw1 + "\n\n" + negative_raw2

        # 시장 배경: market 인덱스 RAG 우선, 없으면 웹 검색 결과로 fallback
        market_store = _get_market_store()
        if market_store:
            market_background, market_sources = run_rag(
                store=market_store,
                query=state["query"],
                prompt_template=MARKET_RAG_PROMPT,
                fallback_msg="시장 리포트 없음",
            )
            logger.info("[Web Search] market 인덱스 RAG 사용")
        else:
            logger.warning("[Web Search] market 인덱스 없음 → 웹 검색 결과로 대체")
            market_background = neutral_raw[:2000]
            market_sources = []

        # Devil's Advocate 프롬프트로 균형 합성
        synthesis_prompt = WEB_SYNTHESIS_PROMPT.format(
            positive=positive_raw[:3000],
            negative=negative_raw[:3000],
            neutral=neutral_raw[:2000],
        )
        llm.invoke([
            SystemMessage(content="당신은 균형 잡힌 산업 분석 전문가입니다. 편향 없이 분석하세요."),
            HumanMessage(content=synthesis_prompt),
        ])

        # 수집된 웹 출처 통합
        web_sources = list(dict.fromkeys(
            src_pos1 + src_pos2 + src_neg1 + src_neg2 + src_neu
        ))
        all_references = (
            [f"[Web] {u}" for u in web_sources] +
            [f"[Market PDF] {s}" for s in market_sources]
        )

        logger.info("[Web Search] 완료")
        return {
            "web_positive":      positive_raw[:4000],
            "web_negative":      negative_raw[:4000],
            "web_neutral":       neutral_raw[:2000],
            "market_background": market_background,
            "references":        all_references,
            "current_step":      "web_search_done",
        }
    except Exception as e:
        logger.error(f"[Web Search] 에러: {e}")
        return {
            "error_log":    [f"web_search: {str(e)}"],
            "current_step": "error",
        }
