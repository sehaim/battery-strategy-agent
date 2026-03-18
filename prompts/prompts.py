MARKET_RAG_PROMPT = """다음 시장 리포트 컨텍스트를 바탕으로 글로벌 배터리·전기차 시장 배경을 분석하세요.

[문서 컨텍스트]
{context}

[분석 요청]
{query}

다음 항목을 포함하여 분석하세요:
1. 글로벌 EV 시장 현황 및 캐즘 여부
2. ESS 시장 성장 동향
3. 배터리 원가 및 기술 트렌드
4. 주요 리스크 및 불확실성

분석 결과:"""

RAG_SYSTEM = """당신은 배터리 산업 전략 분석 전문가입니다.
반드시 제공된 컨텍스트에 있는 정보만 사용하세요.
컨텍스트에 없는 내용은 '해당 정보 없음'으로 명시하세요.
출처(문서명, 페이지)를 가능한 한 명시하세요.
긍정적 내용과 리스크를 균형 있게 서술하세요."""

LG_RAG_PROMPT = """다음 문서 컨텍스트를 바탕으로 LG에너지솔루션의 전략을 분석하세요.

[문서 컨텍스트]
{context}

[분석 요청]
{query}

다음 항목을 포함하여 분석하세요:
1. 포트폴리오 다각화 전략 (ESS, 원통형, 로봇배터리 등)
2. 북미 시장 대응 전략 및 리스크
3. 핵심 기술 경쟁력
4. 재무 현황 및 리스크 요인

분석 결과:"""

CATL_RAG_PROMPT = """다음 문서 컨텍스트를 바탕으로 CATL의 전략을 분석하세요.

[문서 컨텍스트]
{context}

[분석 요청]
{query}

다음 항목을 포함하여 분석하세요:
1. 포트폴리오 다각화 전략 (나트륨이온, ESS, 배터리스왑)
2. 글로벌·신흥시장 공략 전략
3. 원가 경쟁력 및 가격 전략
4. 핵심 기술 경쟁력
5. 리스크 요인

분석 결과:"""

# 확증 편향 방지 - 이중 쿼리
WEB_QUERIES = {
    "lg_positive":  ["LG에너지솔루션 ESS 로봇배터리 성장 전략 성과 2024 2025",
                     "LG Energy Solution new business portfolio success"],
    "lg_negative":  ["LG에너지솔루션 북미 공장 적자 리스크 한계 과제",
                     "LG Energy Solution North America JV losses challenges"],
    "catl_positive":["CATL ESS 나트륨이온 배터리 성장 신흥시장 2024 2025",
                     "CATL sodium battery ESS expansion success"],
    "catl_negative":["CATL 미국 제재 리스크 한계 경쟁 문제점",
                     "CATL US sanctions risks challenges weaknesses"],
    "market":       ["글로벌 전기차 배터리 시장 캐즘 현황 2025",
                     "global EV battery market slowdown outlook 2025"],
}

WEB_SYNTHESIS_PROMPT = """다음은 긍정적 관점과 비판적 관점으로 수집한 정보입니다.
확증 편향 없이 균형 있게 종합하세요.

[긍정 정보]
{positive}

[비판/부정 정보]
{negative}

[시장 배경]
{neutral}

[중요] 긍정적 결론에 반박할 수 있는 근거도 반드시 포함하세요.
한쪽으로 치우친 요약은 허용되지 않습니다.

균형 잡힌 종합 요약:"""

AGGREGATOR_PROMPT = """다음 세 가지 분석 결과를 하나의 통합 데이터로 병합하세요.
중복 내용은 제거하고, 각 출처를 명시하세요.

[LG에너지솔루션 RAG 분석]
{lg_data}

[CATL RAG 분석]
{catl_data}

[웹서치 분석 (편향 방지 적용)]
{web_data}

통합 결과 (출처 명시):"""

SWOT_PROMPT = """다음 통합 분석 데이터를 바탕으로 LG에너지솔루션과 CATL 각각의 SWOT을 생성하세요.
반드시 내부(S/W)와 외부(O/T)를 구분하세요.

[통합 분석 데이터]
{merged_data}

다음 JSON 형식으로 출력하세요:
```json
{{
  "lg": {{
    "S": ["강점1 (출처)", "강점2"],
    "W": ["약점1", "약점2"],
    "O": ["기회1", "기회2"],
    "T": ["위협1", "위협2"]
  }},
  "catl": {{
    "S": ["강점1", "강점2"],
    "W": ["약점1", "약점2"],
    "O": ["기회1", "기회2"],
    "T": ["위협1", "위협2"]
  }}
}}
```"""

INSIGHT_PROMPT = """다음 통합 데이터와 SWOT을 바탕으로 비교 분석과 시사점을 도출하세요.

[통합 데이터]
{merged_data}

[LG SWOT]
{swot_lg}

[CATL SWOT]
{swot_catl}

다음을 포함하세요:
1. 핵심 전략 비교 (사업 / 기술 / 지역 / 리스크 대응 4축)
2. 한국 배터리 산업 시사점
3. 향후 전망"""

REFLECT_PROMPT = """다음 보고서 초안을 Success Criteria 기준으로 검토하세요.

[보고서 초안]
{draft}

[Success Criteria 체크리스트]
C1. SUMMARY + 시장배경 + LG전략 + CATL전략 + SWOT + 시사점 + REFERENCE 전 섹션 포함
C2. SWOT이 내부(S/W) + 외부(O/T) 4분면 표 형식으로 구분
C3. 근거성 — 주요 주장마다 출처 명시
C4. 확증 편향 없이 양사 긍정·비판 균형 서술
C5. SUMMARY 0.5페이지(250단어) 이내

각 항목 PASS/FAIL 판정 후 다음 JSON으로 출력:
```json
{{
  "quality_passed": true/false,
  "feedback": "FAIL 항목과 수정 지시 (PASS면 빈 문자열)"
}}
```"""

REPORT_PROMPT = """다음 분석 데이터를 바탕으로 완성도 높은 전략 분석 보고서를 마크다운으로 작성하세요.

[시장 배경]
{market_background}

[LG 전략]
{lg_strategy}

[CATL 전략]
{catl_strategy}

[비교 분석 및 시사점]
{comparison}
{implications}

[LG SWOT]
{swot_lg}

[CATL SWOT]
{swot_catl}

[Reflect 피드백]
{feedback}

보고서 목차 (반드시 준수):
# SUMMARY  ← 250단어 이내, 핵심 수치·결론·시사점만
# 1. 시장 배경
## 1.1 글로벌 전기차 캐즘 현황
## 1.2 ESS 시장 급성장 및 배터리 수요 다변화
## 1.3 배터리 원가 혁신 트렌드
# 2. LG에너지솔루션 전략 분석
## 2.1 사업 실적 및 현황
## 2.2 포트폴리오 다각화 전략
## 2.3 북미 시장 리스크 및 대응
## 2.4 핵심 경쟁력
# 3. CATL 전략 분석
## 3.1 사업 실적 및 현황
## 3.2 포트폴리오 다각화 전략
## 3.3 신흥시장 공략
## 3.4 핵심 경쟁력
# 4. 전략 비교 및 SWOT 분석
## 4.1 핵심 전략 비교표
## 4.2 LG에너지솔루션 SWOT
## 4.3 CATL SWOT
## 4.4 양사 SWOT 종합 비교
# 5. 종합 시사점
## 5.1 전략적 차이점 요약
## 5.2 한국 배터리 산업 시사점
## 5.3 향후 전망 및 모니터링 지표
# REFERENCE"""

SUMMARY_PROMPT = """다음 보고서의 SUMMARY를 작성하세요.
반드시 250단어 이내로 작성하세요. 개요가 아닌 실질 인사이트 중심입니다.

[보고서 내용]
{report}

SUMMARY (250단어 이내):"""
