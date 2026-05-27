# OER Catalyst SDL — Agentic Self-Driving Lab

CID (Chemical Intent Descriptor) v3.2 기반 자율 합성 실험실 시스템.

## 구조

```
oer-catalyst-project/
├── multiagent/           # AutoGen 멀티에이전트 시스템
│   ├── cid_agent.py      # CID Agent (3에이전트 파이프라인)
│   ├── run_translator.py # 공정 번역기 (6에이전트)
│   ├── run.py            # 기본 프로토콜 생성기
│   ├── web_ui.py         # Gradio 웹 UI
│   └── data_tools.py     # 에이전트 도구 함수 (11개)
├── scripts/              # 데이터 처리 파이프라인
│   ├── 01_extract_sentences.py
│   ├── 02_embed_sentences.py
│   ├── 07_cluster_v3.py
│   ├── 13_contextual_intent.py
│   ├── 14_synthesis_network.py
│   ├── 16_route_performance.py
│   ├── 17_lab_constrained.py
│   ├── 19_scientific_evidence.py
│   └── ...
├── docs/
│   └── CID_v3_research_briefing.md  # CID 스펙 문서
├── output/               # 파이프라인 출력 (git 제외)
├── data/                 # 원시 데이터 (git 제외)
│   └── raw -> (symlink to data)
└── .env                  # API 키 (git 제외)
```

## 설치

```bash
pip install pyautogen autogen-ext[openai] openai gradio pandas numpy scikit-learn sentence-transformers umap-learn hdbscan plotly
```

## 설정

```bash
# .env 파일 생성
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

## 실행

```bash
# CID 생성 (논문 → CID 6컬럼 번역)
python multiagent/cid_agent.py "NiFe LDH"

# 공정 번역 (불가능한 공정 → 가능한 공정)
python multiagent/run_translator.py "NiFe LDH, η < 260 mV"

# 웹 UI
python multiagent/web_ui.py
# → http://localhost:7860
```

## CID v3.2 스펙

6컬럼 구조:
1. **Raw Step** — 논문 원문 그대로
2. **Chemical Intent** — 10개 닫힌 집합 (Nucleation, Crystallization, Redox Control, Stoichiometry Control, Doping, Morphology Control, Purification, Drying, Catalyst-Electrode Coupling, Electrochemical Activation)
3. **Mechanism** — 열린 집합 (원인→과정→결과 체인)
4. **Tunable Conditions** — 조절 가능한 변수 + 논문 값
5. **Required Capabilities** — 필요한 장비/기능
6. **Output Form** — 산물 형태

## SDL 아키텍처

```
Query Parser → Literature Agent → CID Agent → Experimental Agent → BO
```

## 라이선스

MIT
