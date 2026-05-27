# CID (Chemical Intent Descriptor) v3.1 — Research Briefing

> 최종 업데이트: 2026-05-27  
> 현재 버전: v3.1 확정, v3.2 후보 변경사항 합의 완료

---

## 1. 연구 목표와 배경

최종 목표는 **Agentic Self-Driving Laboratory(SDL)**를 만드는 것이다.
SDL은 문헌에서 합성 절차를 읽고, 실험을 설계하고, 로봇이 실행하고, 결과를 측정해서 다시 최적화하는 자율 실험 루프다.

이 루프가 작동하려면 "논문에 적힌 합성 절차"를 기계가 이해할 수 있는 형태로 번역해야 한다. 그 번역의 결과물이 **CID(Chemical Intent Descriptor)**다.

**CID의 핵심 가치**: 같은 물질을 만들더라도 "왜 이 step을 하는가"와 "어떤 메커니즘으로 달성하는가"는 공정마다 다르고, 그 차이를 잡아내는 것이 CID의 존재 이유다.

주요 예시 시스템: **NiFe OER 촉매** (NiFe-LDH). 최소 7가지 서로 다른 합성법으로 만들 수 있어서 CID의 가치를 증명하기에 적합하다.

---

## 2. SDL 전체 아키텍처와 CID의 위치

```
사용자 쿼리 (자연어)
  │
  ▼
[ Query Parser ]
  "Nanosheet 형태의 η@10 < 220 mV인 NiFe-LDH 만들어줘"
  │
  ▼
structured target spec
  target = (구조, 성능)
    구조 = (조성: NiFe-LDH, 상: LDH, 형태: nanosheet)
    성능 = (η@10 < 220 mV)
  │
  ▼
[ Literature Agent ] ── 논문에서 합성 문단 추출
  │
  ▼
[ CID Agent ] ── 합성 문단 → CID로 번역 (사실만, 판단 없음)
  │
  ▼
  CID (Chemical Intent Descriptor)
  ┌─────────────────────────────────────────────────┐
  │ Raw Step │ Intent │ Mechanism │ Conditions │ ... │
  │ 1행 = 1 Intent. 판단 없이 사실만 기록.            │
  └─────────────────────────────────────────────────┘
  │
  ▼
[ Experimental Agent ] ── CID를 읽고 판단
  • 이 메커니즘을 우리 장비로 구현 가능한가?
  • 대체 가능한 시약/조건이 있는가?
  • 공정을 어떤 순서로 조립할 것인가?
  │
  ▼
[ Bayesian Optimization (BO) ]
  CID의 Tunable Conditions를 받아서 세부 조건 최적화
  │
  ▼
  실험 실행 → 측정 → 동일한 target spec으로 검증
```

**핵심 설계 원칙**: 입력 spec = 검증 spec

두 가지 운전 모드:
- **Reproduction 모드**: spec이 좁다 → 특정 논문의 공정을 재현
- **Discovery 모드**: spec이 넓다 → 다중 CID 후보를 생성하고 BO가 탐색

---

## 3. CID v3.1 스펙

### 3.1 정의
CID는 논문의 합성 절차를 읽고 변환한 구조화된 기술(descriptor). 이것만 보고 다시 실행 가능한 공정으로 변환할 수 있어야 한다. **사실만 기술하고 판단은 하지 않는다.**

### 3.2 6개 컬럼 구조

| 컬럼 | 내용 | 성격 |
|------|------|------|
| Raw Step | 논문 원문 그대로 | 정보 손실 방지 |
| Chemical Intent | "이 step은 ○○을 하기 위한 것이다"에서 ○○ | 목적 (10개 닫힌 집합) |
| Mechanism | 어떤 화학적 메커니즘으로 그 목적을 달성하는가 | 수단 (열린 집합) |
| Tunable Conditions | 조절 가능한 변수 + 논문 값 | 사실만 |
| Required Capabilities | 필요한 장비 기능 | 자연어 |
| Output Form | 이 step 산물의 형태 | 다음 step 연결용 |

### 3.3 Chemical Intent 닫힌 집합 (10개)

1. **Nucleation** — 핵형성
2. **Crystallization** — 결정화
3. **Redox Control** — 산화환원 제어
4. **Stoichiometry Control** — 화학양론 제어
5. **Doping** — 도핑
6. **Morphology Control** — 형태 제어
7. **Purification** — 정제
8. **Drying** — 건조
9. **Catalyst-Electrode Coupling** — 촉매-전극 결합
10. **Electrochemical Activation** — 전기화학적 활성화

### 3.4 핵심 규칙

1. **1행 = 1 Intent**, 예외 없음
2. Mechanism은 열린 집합. "왜 이 조건이 필요한지 화학적으로 무슨 일이 일어나는지"를 기술
3. Tunable Conditions은 변수 + 논문 값만. 판단(criticality)은 넣지 않음
4. Required Capabilities는 자연어로 느슨하게. LLM이 의미적 매칭
5. 합성 방법 라벨(hydrothermal, sol-gel)은 CID에 포함하지 않음

### 3.5 분류 규칙

- Function은 "이 step은 ○○을 하기 위한 것이다"에서 ○○
- 같은 ○○을 다른 방법으로 달성 = 서로 다른 Mechanism
- Vacancy, local ordering, anion exchange = 현상(phenomenon), Intent 아님
- Hydrothermal, sol-gel = 합성 방법 라벨, CID에 포함하지 않음

### 3.6 적용 예시

**Raw Step**: Substituted rutile materials were synthesized using potassium perruthenate and metal peroxide or oxide in a molar ratio Ru:M of 1:0.5 in 10 mL of H₂O, based on 2.45 mmol of Ru

| 컬럼 | 값 |
|------|-----|
| Chemical Intent | Stoichiometry Control |
| Mechanism | Ru/M precursor를 수용액에서 혼합 → 원자 수준 접촉 유도, 이후 열처리 시 rutile lattice 내 M 치환 가능한 조성적·공간적 혼합 |
| Tunable Conditions | Ru:M=1:0.5, Ru=2.45mmol, Vol=10mL |
| Required Capabilities | liquid mixing, dissolving |
| Output Form | solution (Ru/M precursor mixture in H₂O) |

---

## 4. "같은 물질 다른 합성법" — NiFe LDH 7가지 공정

| 공정 | Mechanism 핵심 차이 | OER 성능 |
|------|---------------------|----------|
| RT epoxide route | Epoxide가 H⁺를 받아 OH⁻를 천천히 방출 → 균일 핵형성 | Best (RT > HT) |
| Hydrothermal | Urea/NH4F 가수분해 → OH⁻ 서서히 공급 + F⁻ capping | 경우에 따라 RT보다 높음 |
| Co-precipitation | NaOH 직접 첨가 → 즉시 supersaturation | 230 mV @10 |
| Mechanochemical | Mechanical force로 solid-state 반응 | 221 mV @10 |
| Electrodeposition | Cathodic OH⁻ 발생 → 기판 표면 직접 핵형성 | Fe-rich 영역 도달 가능 |
| RT green synthesis | NF 표면에서 에너지 입력 없이 직접 성장 | 181 mV @10, Tafel 42.3 |
| HT + NaBH₄ 후처리 | NaBH₄가 Fe³⁺ 주변 vacancy 생성 + Ni³⁺ enrichment | N/A |

**핵심**: Intent 집합은 7개 사례에서 상당 부분 겹치지만, Mechanism이 모두 다름.
OH⁻ 공급만 봐도: epoxide proton scavenging / urea hydrolysis / 직접 base 첨가 / cathodic generation으로 분기.

---

## 5. "같은 공정, 다른 Intent" 반박 사례

Urea 한 분자의 세 가지 얼굴:
- NH4F와 함께 hydrothermal → **Nucleation** (OH⁻ source via hydrolysis)
- 단독 + Ni acetate, hydrothermal → **Nucleation + Morphology Control** (OH⁻ + vertical nanosheet 성장 유도)
- Ru/M nitrate 수용액에서 → **Stoichiometry Control** (combustion fuel, OH⁻ source가 아님)

같은 hydrothermal 공정, 같은 urea 분자인데 공존 이온과 농도가 바뀌면 Intent가 분기한다.
→ 공정 라벨 분류로는 절대 잡히지 않는 "의도/메커니즘 분기"를 잡는 것이 CID의 가치.

---

## 6. 재현성 정의 — "구조 + 성능"

### 구조(Structure) 검증
- 조성(T0): ICP 측정, 합격 기준 ±5%
- 상(T1): XRD 측정, 합격 기준 2θ shift < 0.2°
- 형태(T2): SEM 측정, 합격 기준 같은 morphology family

### 성능(Performance) 검증
- OER 활성: LSV 측정, η@10 ±20 mV
- Tafel slope: ±10 mV/dec
- 안정성: 24h CP에서 η 증가 < 30 mV

---

## 7. v3.2 후보 변경사항

| 구 항목 | 신 항목 | 비고 |
|---------|---------|------|
| Chemical Function | Chemical Intent | 의도에 해당 |
| Driving Force | Mechanism | 메커니즘 수준 기술 강제 |
| Control Levers | Tunable Conditions | BO가 tuning하는 변수 |

---

## 8. 검증 데이터 현황

- CID 추출 완료: 49개 OER 논문에서 step-level CID 735행
- 시나리오 1: NiFe LDH × 7개 공정, 13건 문헌
- 시나리오 2: 같은 hydrothermal 공정 내 분기 사례 8건

---

## 9. 미해결 과제

| 옵션 | 작업 | 예상 시간 |
|------|------|-----------|
| A | CID_v3_spec.md를 v3.2로 업데이트 | 1 세션 |
| B | 자연어 → structured target spec 파서 | 1 세션 |
| C | 외부 LLM으로 추가 사례 발굴 | 1 세션 + LLM |
| D | CID Agent Mechanism 추출 품질 eval set | 2~3 세션 |
| E | 다른 target family 발굴 (NiFe₂O₄, NiCo LDH) | 1 세션 |
