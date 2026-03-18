import os
import logging
from collections import defaultdict
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from prompts.prompts import WEB_QUERIES, WEB_SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)

MAX_PER_DOMAIN = 2


def _get_search_tool():
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        return TavilySearchResults(max_results=5)
    except Exception:
        from langchain_community.tools import DuckDuckGoSearchRun
        logger.warning("Tavily 없음 → DuckDuckGo fallback")
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


def _search(tool, queries: List[str]) -> str:
    """여러 쿼리 실행 후 중복 제거하여 텍스트 반환"""
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

    texts = []
    for r in filtered:
        if isinstance(r, dict):
            content = r.get("content", r.get("snippet", str(r)))
            url = r.get("url", "")
            texts.append(f"{content}\n출처: {url}" if url else content)
        else:
            texts.append(str(r))
    return "\n\n".join(texts) or "검색 결과 없음"


def web_search_node(state: AgentState) -> dict:
    logger.info("[Web Search] 시작 — 이중 쿼리 편향 방지")
    try:
        tool = _get_search_tool()
        llm  = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0.1)

        positive_raw = (
            _search(tool, WEB_QUERIES["lg_positive"]) + "\n\n" +
            _search(tool, WEB_QUERIES["catl_positive"])
        )
        negative_raw = (
            _search(tool, WEB_QUERIES["lg_negative"]) + "\n\n" +
            _search(tool, WEB_QUERIES["catl_negative"])
        )
        neutral_raw = _search(tool, WEB_QUERIES["market"])

        # Devil's Advocate 프롬프트로 균형 합성
        synthesis_prompt = WEB_SYNTHESIS_PROMPT.format(
            positive=positive_raw[:3000],
            negative=negative_raw[:3000],
            neutral=neutral_raw[:2000],
        )
        synthesized = llm.invoke([
            SystemMessage(content="당신은 균형 잡힌 산업 분석 전문가입니다. 편향 없이 분석하세요."),
            HumanMessage(content=synthesis_prompt),
        ]).content

        logger.info("[Web Search] 완료")
        return {
            "web_positive":      positive_raw[:4000],
            "web_negative":      negative_raw[:4000],
            "web_neutral":       neutral_raw[:2000],
            "market_background": neutral_raw[:2000],
            "current_step":      "web_search_done",
        }
    except Exception as e:
        logger.error(f"[Web Search] 에러: {e}")
        return {
            "error_log":    [f"web_search: {str(e)}"],
            "current_step": "error",
        }
