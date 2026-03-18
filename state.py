from typing import TypedDict, List, Optional, Annotated
import operator


class AgentState(TypedDict):
    query: str

    # 수집 결과
    lg_rag_result:     Optional[str]
    catl_rag_result:   Optional[str]
    web_positive:      Optional[str]
    web_negative:      Optional[str]
    web_neutral:       Optional[str]
    market_background: Optional[str]

    # 분석 결과
    merged_data:  Optional[str]
    swot_lg:      Optional[dict]
    swot_catl:    Optional[dict]
    comparison:   Optional[str]
    implications: Optional[str]
    draft_report: Optional[str]
    final_report: Optional[str]
    references:   List[str]

    # Reflect
    quality_passed:    bool
    revision_feedback: Optional[str]
    revision_count:    int

    # 에러 핸들링
    error_log:          Annotated[List[str], operator.add]
    retry_count:        int
    max_retries:        int
    current_step:       str
    fallback_triggered: bool
    is_complete:        bool
