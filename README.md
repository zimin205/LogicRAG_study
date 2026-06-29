# LogicRAG — 연구용 정리본

> 원본 레포지토리: [chensyCN/LogicRAG](https://github.com/chensyCN/Agentic-RAG) | 논문: [AAAI 2026](https://openreview.net/forum?id=ov1bwU35Mf) / [Arxiv](https://arxiv.org/abs/2508.06105)

## 개요

이 폴더는 **GraphRAG vs. LogicRAG 모델 비교 연구**를 위해 원본 LogicRAG 코드를 클론 후 실험에 맞게 구성한 것입니다.

LogicRAG는 코퍼스에 지식 그래프를 사전 구축하지 않고, **질문 자체를 논리 의존 그래프(Query Logic DAG)로 분해**하여 단계적 검색을 수행하는 RAG 시스템입니다. GraphRAG가 인덱싱 단계에서 그래프를 만드는 것과 달리, LogicRAG는 추론 시점(test-time)에 동적으로 구조를 생성합니다.

## 폴더 구조

```
LogicRAG/
├── .env                          # API 키 (gitignore 처리됨)
├── config/config.py              # 모델·API·임베딩 설정
├── run.py                        # 실행 진입점
│
├── src/
│   ├── main.py                   # 평가 오케스트레이션 (설정은 config.py에서)
│   ├── models/
│   │   ├── base_rag.py           # 임베딩 기반 retrieval 공통 기반 클래스
│   │   ├── logic_rag.py          # LogicRAG 메인 파이프라인 (6단계)
│   │   ├── query_logic_dag.py    # Query Logic DAG 자료구조 및 빌더
│   │   ├── verify_non_cyclicity.py  # DAG 사이클 검증 + 위상 정렬
│   │   ├── dag_topological_rank.py  # 위상 rank 계산
│   │   └── dag_rank_resolver.py  # rank 단위 retrieval + Dynamic Adaptation
│   ├── evaluation/evaluation.py  # 평가 루프, 체크포인트, 메트릭 집계
│   └── utils/utils.py            # OpenAI 호출, JSON 파싱, 정규화 유틸
│
├── dataset/                      # 벤치마크 데이터셋
│   ├── musique.json              # 평가 질문 (1,000개)
│   └── musique_corpus.json       # 검색 코퍼스 (11,656개)
│
├── cache/                        # 코퍼스 임베딩 캐시 (자동 생성)
└── evaluation/                   # 평가 결과 및 체크포인트 (자동 생성)
    ├── checkpoints/              # 5문항 간격 중간 저장
    └── evaluation_results_musique_YYYYMMDD_HHMMSS.json  # 최종 결과
```

## 사전 준비

루트 `.env` 파일에 API 키를 설정합니다.

```
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://...   # 커스텀 엔드포인트 사용 시
HF_TOKEN=your_hf_token_here   # huggingface.co/settings/tokens (Read 권한)
WANDB_API_KEY=your_wandb_api_key_here
```

의존성 설치:

```bash
pip install -r requirements.txt
```

모델·API 설정 변경은 `config/config.py`에서 합니다.

| 설정 | 기본값 | 비고 |
|------|--------|------|
| `DEFAULT_MODEL` | `gpt-4o-mini` | 사용하는 API 엔드포인트에 맞게 수정 |
| `DEFAULT_MAX_TOKENS` | `500` | 응답 잘림 방지용 여유값 |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | 로컬 HuggingFace 모델 (CUDA·MPS·CPU 자동 감지) |
| `CALLS_PER_MINUTE` | `20` | API 속도 제한 |

## 실험 실행 방법

실험 세팅은 `config/config.py` 하단의 **Experiment Configuration** 섹션에서 변경합니다.

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `LIMIT` | `1000` | 평가할 질문 수 (`0` = 전체) — 전체 실행 전 `5`로 샘플 테스트 |
| `TOP_K` | `3` | 한 번에 검색할 context 수 (논문: k=3) |
| `MAX_ROUNDS` | `5` | Dynamic DAG Adaptation 최대 횟수 (논문: 5) |
| `CHECKPOINT_INTERVAL` | `5` | 체크포인트 저장 간격 |

설정 변경 후 실행:

```bash
python run.py
```

## 파이프라인 개요

LogicRAG는 질문 하나에 대해 아래 6단계를 순차 실행합니다.

| 단계 | 설명 |
|------|------|
| Stage 1 | **Query Decomposition** — 질문을 서브문제로 분해 (few-shot prompting) |
| Stage 2 | **DAG 구성** — 서브문제 간 논리 의존 관계를 그래프로 구성 |
| Stage 3 | **사이클 검증 + repair** — DAG 무결성 확인, 사이클 발생 시 LLM으로 자동 수정 |
| Stage 4 | **위상 정렬 + rank 계산** — 처리 순서를 rank 단위로 그룹화 |
| Stage 5 | **Rank 단위 Retrieval + Dynamic Adaptation** — 부모 답변을 조건으로 검색, 필요 시 서브문제 동적 추가 |
| Stage 6 | **최종 답 합성** — 모든 서브 답변을 종합해 최종 답 생성 |

## 데이터셋

| 데이터셋 | 질문 유형 | QA 수 | Corpus 수 |
|---------|---------|------|---------|
| MuSiQue | 2~4-hop | 1,000 | 11,656 |

비교 모델: [GraphRAG (Microsoft)](https://github.com/microsoft/graphrag)
