"""RAG 에이전트 공통 로직"""
import os
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from retrieval.pdf_loader import load_pdfs, chunk_pages
from retrieval.embedder import FAISSVectorStore
from prompts.prompts import RAG_SYSTEM

logger = logging.getLogger(__name__)


def get_llm():
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0.1,
    )


def build_store(data_dir: str, cache_path: str) -> FAISSVectorStore | None:
    """벡터스토어 로드 또는 구축"""
    if os.path.exists(f"{cache_path}.faiss"):
        try:
            store = FAISSVectorStore.load(cache_path)
            logger.info(f"캐시 로드: {cache_path}")
            return store
        except Exception as e:
            logger.warning(f"캐시 로드 실패: {e}")

    pages = load_pdfs(data_dir)
    if not pages:
        logger.warning(f"{data_dir}/ 에 PDF 없음")
        return None

    chunks = chunk_pages(pages)
    store = FAISSVectorStore(chunks)
    try:
        store.save(cache_path)
    except Exception as e:
        logger.warning(f"캐시 저장 실패: {e}")
    return store


def run_rag(store: FAISSVectorStore | None, query: str, prompt_template: str,
            fallback_msg: str) -> str:
    """RAG 검색 후 LLM 분석 실행"""
    if store:
        results = store.search(query)
        context = store.format_context(results)
    else:
        context = fallback_msg

    llm = get_llm()
    prompt = prompt_template.format(context=context, query=query)
    response = llm.invoke([
        SystemMessage(content=RAG_SYSTEM),
        HumanMessage(content=prompt),
    ])
    return response.content
