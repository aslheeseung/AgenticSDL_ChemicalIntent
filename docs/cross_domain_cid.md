# Cross-Domain CID: 타 도메인의 Intent/Mechanism을 활용한 공정 혁신

## 개념

기존 CID: 같은 물질(NiFe LDH)의 여러 합성법을 비교
Cross-Domain CID: **완전히 다른 도메인에서 같은 Intent를 달성하는 Mechanism을 가져와서 적용**

핵심 질문: "이 Intent를 달성하는 다른 방법이, 완전히 다른 분야에 이미 존재하지 않는가?"

---

## OER 905편에서 발견한 Cross-Domain 사례

### 사례 1: MOF → OER 촉매 (가장 강력)

**도메인 A**: MOF (Metal-Organic Framework) — 가스 저장/분리 용도로 개발된 다공성 물질
**도메인 B**: OER 촉매 — 전기화학적 물 분해용 전극

**연결 Intent**: Crystallization + Morphology Control

```
도메인 A (MOF)의 Mechanism:
  "유기 리간드(2-methylimidazole)가 금속 이온(Co²⁺/Zn²⁺)과 
   배위결합 → 3차원 다공성 격자 형성 → 
   열분해 시 리간드가 탄소로 변환되면서 금속-탄소 복합체 생성"

도메인 B (OER)에 적용:
  "ZIF-67(Co-MOF)를 합성 → 열처리 → 
   다공성 Co-N-C 구조가 OER 활성점으로 작용"
  
  논문: Fe₂O₃@ZIF-67 core-shell → sulfidation → Fe₂O₃@h-Co₉S₈@C yolk-shell
  Intent: Nucleation (MOF의 시드 매개 성장을 OER 촉매에 적용)
```

**CID로 표현**:
| 컬럼 | 값 |
|------|-----|
| Chemical Intent | Nucleation |
| Mechanism | MOF 리간드가 금속 이온과 배위 → 3D 다공성 네트워크 형성 → 열분해로 리간드 제거 시 균일한 나노입자 분산 |
| Cross-Domain Origin | MOF (가스 분리) → OER (전기화학) |

### 사례 2: Soft Template (Block Copolymer) → OER 촉매

**도메인 A**: Polymer Science — 블록 공중합체 자기조립
**도메인 B**: OER 촉매 — 중공 금속 황화물

```
논문: "hollow metal sulfide heterostructures prepared via self-assembly 
       using PS-b-PAA-b-PEG as soft template"

도메인 A의 Mechanism:
  "PS-b-PAA-b-PEG 삼중 블록 공중합체가 용매 중에서 미셀 자기조립 →
   소수성 코어/친수성 쉘 구조 → 
   금속 이온이 친수성 영역에 선택적 흡착"

도메인 B에 적용:
  "소수성 코어를 금속 전구체로 채우고 → 열분해로 중합체 제거 →
   중공 구조의 금속 황화물 형성 (높은 비표면적 = 높은 OER 활성)"
```

### 사례 3: Hard Template (SBA-15 Zeolite) → OER 촉매

**도메인 A**: Zeolite Science — 분자체 촉매
**도메인 B**: OER 촉매

```
논문: "Zeolite SBA-15 as hard template, unique structure and fine 
       hydrothermal stability"

Intent: Morphology Control
Mechanism: "SBA-15의 규칙적인 메조포어(5-30nm)에 전구체 주입 → 
           열처리 후 SBA-15를 NaOH로 제거 → 
           메조포어의 역 복제본인 규칙적 다공성 구조 생성"
Cross-Domain: Zeolite (석유화학 촉매) → OER (전기화학)
```

### 사례 4: Sacrificial Template (ZIF → Metal Sulfide)

**도메인 A**: MOF 합성 → **도메인 B**: 고엔트로피 합금 황화물

```
논문: "FeCoNiCu-MOF template → carbonization → (FeCoNiCu)S₂ sulfides"

Intent: Stoichiometry Control + Crystallization
Mechanism: "MOF 내 금속 이온이 이미 원자 수준에서 균일하게 분산 →
           열분해 시 이 분산성이 유지되어 고엔트로피 합금 형성 →
           황화 시 각 금속이 균일하게 치환"
Cross-Domain: MOF (정밀 화학) → High-Entropy Alloy (신소재)
```

### 사례 5: Coagulation-inspired (Colloid Science → Catalyst)

```
논문: "Ag@hydroxides composites was initially inspired by the 
       coagulation of mixed colloids"

Intent: Nucleation
Mechanism: "은 콜로이드와 수산화물 콜로이드의 혼합 →
           전하 중성화에 의한 공침(coagulation) →
           Ag가 수산화물 매트릭스 내에 균일 분산"
Cross-Domain: Colloid Science (물리화학) → OER 복합 촉매
```

---

## Cross-Domain CID의 패턴 정리

| 패턴 | 도메인 A | 도메인 B | 연결 Intent |
|------|---------|---------|-------------|
| MOF-derived | 가스 저장/분리 | OER 촉매 | Crystallization, Morphology Control |
| Soft template | Polymer Science | OER 촉매 | Morphology Control |
| Hard template | Zeolite Science | OER 촉매 | Morphology Control |
| Sacrificial template | MOF | 고엔트로피 합금 | Stoichiometry Control |
| Coagulation | Colloid Science | 복합 촉매 | Nucleation |

**공통점**: Morphology Control과 Nucleation이 Cross-Domain에서 가장 많이 활용되는 Intent.

---

## CID 시스템에 Cross-Domain 검색을 추가하는 방법

### 방법 1: Intent 기반 Cross-Domain 검색
```
1. CID에서 특정 Intent(예: Morphology Control)를 식별
2. 같은 Intent를 가진 다른 재료/방법의 Mechanism을 검색
3. "이 Mechanism을 현재 문제에 적용할 수 있는가?" 평가
```

### 방법 2: Problem → Solution 매칭
```
1. 현재 합성의 문제를 Intent로 표현
   예: "결정성이 낮다" → Crystallization Intent의 Mechanism 불충분
2. 다른 도메인에서 같은 문제를 해결한 Mechanism 검색
   예: MOF에서 사용하는 "리간드 배위 후 열분해" Mechanism
3. 적용 가능성 평가
```

### 방법 3: LLM 기반 Analogy 검색
```
1. LLM에게 "이 Mechanism과 유사한 화학적 원리를 사용하는 
   다른 분야의 합성법을 제안해주세요"라고 질문
2. 제안된 방법을 CID로 분해
3. 실험실 제약 내에서 실행 가능한지 검증
```

---

## 다음 단계

1. **Cross-Domain 검색 에이전트** 구현
   - Intent + Mechanism을 입력하면 다른 도메인의 사례를 검색
   - 현재 905편 데이터로 프로토타입

2. **Web Search 연동** (선택)
   - 외부 논문 데이터베이스에서 실시간 검색
   - Semantic Scholar, arXiv API 활용

3. **평가**
   - Cross-Domain CID가 실제로 새로운 프로토콜을 생성하는지
   - 생성된 프로토콜의 화학적 타당성
