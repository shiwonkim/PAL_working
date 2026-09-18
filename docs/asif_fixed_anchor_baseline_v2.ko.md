# Fixed vs Learnable Anchors — Rebuttal 전략 · 개념 정리 · 범위 갱신 (v2)

> **이 문서의 위치.** `docs/asif_fixed_anchor_baseline.ko.md`(v1)는 **구현 설계**(레이어/
> 트레이너/config/ablation grid/코드 체크리스트)이고 **이미 구현 완료**됐다. 이 v2는 그 위에서
> 진행한 **리뷰 대응 전략 · 개념(ASIF 원리, prototype) · 실행 범위 갱신**을 담는다. 구현
> 세부는 v1 참조. 상충 시 개념·범위는 v2, 코드는 v1.
>
> 상태: NeurIPS 2026 rebuttal. 실험 = **fixed anchors(ASIF) vs learnable anchors(PAL)**.

---

## 1. ASIF는 어떻게 "학습 없이" 정렬하나 — paired-anchor 공유 좌표계

frozen 인코더 `f_I`, `f_T`는 좌표계가 완전히 다르다. ASIF(Norelli et al., NeurIPS'23)는
**학습 없이** 이를 잇는다. 비결 = **앵커가 image-text *쌍***이라는 것.

N개 쌍 앵커 `{(x₁,y₁),…,(x_N,y_N)}`에 대해:
- 이미지 `x`의 상대표현: `r_I(x) = [sim(f_I(x), f_I(xᵢ))]ᵢ ∈ ℝ^N`
- 텍스트 `y`의 상대표현: `r_T(y) = [sim(f_T(y), f_T(yᵢ))]ᵢ ∈ ℝ^N`

**앵커 i의 이미지 xᵢ와 텍스트 yᵢ가 같은 쌍**이라, `r_I`의 i번째와 `r_T`의 i번째가 "**앵커 개념
i와 얼마나 닮았나**"로 **같은 의미**를 갖는다 → 두 상대표현이 **같은 N차원 공간(축=앵커)**에
산다 → **그냥 코사인 비교하면 cross-modal 비교**가 된다. 즉 **정렬을 "학습"하는 게 아니라
앵커의 *페어링*이 정렬을 *유도***한다.

- 정제 장치 2개(pal.py의 `topk`/`sim_exponent`): **top-k sparsify**(각 상대표현에서 상위 k개만
  남기고 0 — 샘플별 노이즈 꼬리 제거), **exponent p**(남은 유사도 `val^p`, 강한 매칭 sharpen).
- eval: zero-shot 분류 = 이미지 상대표현 vs 클래스-프롬프트 상대표현을 같은 앵커공간에서
  argmax. retrieval = 상대표현끼리 유사도 랭킹. **학습 파라미터 0개.**

---

## 2. PAL vs ASIF — 결정적 차이 = "대응을 어디서 얻나"

| | ASIF (fixed) | PAL (learnable) |
|---|---|---|
| 앵커 | 데이터 쌍으로 **고정** | 모달리티별 K개 **학습** |
| i축의 image↔text 대응 | **페어링으로 *주어짐*** (공짜) | **contrastive loss로 *학습*** |
| 앵커 수 | 보통 큼(커버리지) | 작게(K=512) 최적화 |
| pooling | pooled 코사인 (토큰 개념 없음) | token-level **CAP** |

→ **PAL = ASIF의 "앵커가 페어여야 한다"는 제약을 *학습*으로 대체.** 이 실험은 그 "학습"의
가치를 격리·정량화한다. (PAL의 novelty = relative-rep 계보의 **"learning" 스텝**.)

---

## 3. ⭐ 두 축의 구분 (갱신된 이해)

혼동 주의: **데이터 풀 크기**와 **앵커 구성 방법**은 완전히 별개 축이다.

| 축 | 무엇 | 변형 |
|---|---|---|
| **축 A: 데이터 풀 n** | 앵커를 뽑는 풀 크기 | n=1k/2k/…/50k |
| **축 B: 앵커 구성 방법** | K개를 어떻게 정하나 | random / ASIF-sample / kmeans / **learned** |

### 축 A의 함정 — fixed-K로 n만 늘리면 ≈ **시드 노이즈**
K=512 고정 시, n=1k든 50k든 결국 **512개를 랜덤 i.i.d.로 뽑는 것** → 앵커셋의 통계적 품질은
(n≥K인 한) 동일 → **ASIF는 n에 대해 flat**, 흔들림은 시드 노이즈. **랜덤 샘플링은 더 좋은
앵커를 *고르지* 않으므로**, 큰 풀이 도움 안 됨.
→ **fixed-K를 전 n-격자로 돌릴 필요 없음.** ASIF는 **flat 기준선 1개(+시드 밴드)**로 충분.

### 의미 있는 스케일링 = **K=n** (ASIF가 데이터를 *실제로* 쓰는 유일한 길)
ASIF가 데이터를 활용하는 방법 = 앵커를 더 쓰기(K↑). 그래서:
- **PAL(K=512) across n** → 상승 (학습으로 데이터 활용; 이미 보유: 곡선 0.097→0.563)
- **ASIF(K=n, 전부 앵커) across n** → 상승하되 PAL보다 낮음 → "ASIF에 데이터를 *최대로* 줘도
  PAL이 이김"
- **2Mov-W3 직답**: "데이터 더 주면 fixed/contrastive도 좋아질 텐데 얼마나?" → fixed-K는 거의
  안 좋아짐(flat), K=n도 PAL만큼 못 오름.

(제약: 트레이너에 `K > n_train이면 에러`. n=500이면 K≤500.)

---

## 4. 축 B — 앵커 구성 방법 사다리 (+ pairing 제약)

같은 데이터·K에서 **앵커를 어떻게 얻느냐**만 바꾼다:

| 변형 | 앵커 | 페어링 | 예상 |
|---|---|---|---|
| **fixed-random** | 랜덤 벡터 (데이터 아님) | ❌ 없음 | **붕괴** — 고정+비페어는 정렬 불가 (floor) |
| **fixed-ASIF** | 데이터에서 **샘플한 실제 쌍** | ✅ 데이터 | Z (topk/p로 강화) — 진짜 ASIF |
| **fixed-kmeans** | 임베딩 **클러스터 중심** | ⚠️ 조건부 | Z' — 단 **쌍 함께 클러스터링**해야 대응 보존 |
| **PAL** | **학습된** 벡터 | ✅ 학습 | 최고 |

- **pairing 필수**: image·text 앵커를 *따로* 만들면(random, 분리-kmeans) i축 대응이 없어 붕괴.
  ASIF는 데이터 페어링으로, PAL은 학습으로 대응을 확보. kmeans는 쌍을 함께 클러스터링해
  (한 모달리티로 클러스터→같은 인덱스 파트너) 페어링을 살려야 공정.
- **⚠️ v1의 중요 관측**: `fixed_cap`(고정앵커 + **CAP**)은 **near-chance(val≈7.98)**. CAP의
  softmax pooling은 *학습된* query 앵커를 요구하므로, **fixed/CAP은 "CAP은 학습 앵커에
  의존한다"는 ablation**이지 ASIF의 토큰판이 아니다. → **작동하는 fixed baseline은 pooled(CLS)
  레벨**(fixed/pooled=ASIF-등가). 사다리 비교는 **pooled 레벨**에서, PAL-CAP은 헤드라인으로.

---

## 5. "PAL은 결국 prototype-based alignment 아닌가?" — 정직한 인정 + 재정의

### prototype은 라벨 없이도 구한다 (3방식)
| 방식 | prototype을 어떻게 | 라벨 |
|---|---|---|
| ① 클래스 평균 (ProtoNet) | 클래스별 feature 평균 | ✅ |
| ② 클러스터링 (k-means) | 임베딩 클러스터 **중심** | ❌ |
| ③ **학습 파라미터** (SwAV/DINO/slot-attn) | 랜덤 init 벡터 K개를 **목적함수로 학습** | ❌ |

prototype-based alignment = **feature를 K개 prototype에 대한 (소프트) 할당으로 표현하고 그
할당 공간에서 정렬**. PAL의 앵커 = ③(학습 파라미터형 prototype), 상대표현 = 소프트 할당.

### 정직한 답: **네, 구조적으로 PAL은 learnable-prototype/relative-rep 계열의 일종이다.**
- 부정하면 진다(리뷰어가 이미 간파). **"projection-free"도 엄밀히는 약한 주장** —
  anchor-similarity 매핑 자체가 학습된 feature map(일종의 projection) → 9q1o처럼 *"제한적
  의미에서만 성립"* 인정.
- **하지만 결함 아님.** 대부분 논문은 알려진 메커니즘의 특정 instantiation. 문제는 novelty를
  *메커니즘* 수준에서 주장한 것(→ W1/W2)이지, prototype 계열인 것 자체가 아니다.

### 기여의 재정의 (메커니즘 → instantiation + 경험적 발견)
1. **학습이 핵심이라는 발견**: 같은 prototype 구조라도 `random(비페어) 붕괴 < fixed-ASIF(샘플)
   ≈ kmeans(클러스터) < PAL(학습)` → **fixed-vs-learnable 실험이 정량 입증** (W1 답).
2. **용도가 다름**: SwAV/DINO prototype = 단일모달 SSL 클러스터링. PAL = **frozen 두 공간을 잇는
   cross-modal 정렬**에 modality-specific 학습 prototype → 설계공간의 **새 점**.
3. **token-level CAP + 파라미터 효율**(anchor만 학습) — 저데이터에서 작동하게 하는 구체 설계.

→ 한 줄: **"PAL은 prototype 계열이 맞다. 단, 이 세팅(frozen·저데이터 정렬)에선 prototype을
*학습*하는 게 sample/cluster보다 결정적이고, 그걸 실증한 것이 기여다."** (2Mov가 Originality
4를 준 만큼, 부정 평가는 "차별화 부족"이지 결함이 아니다 → 서술 재정의 + 통제 실험으로 해결.)

---

## 6. 리뷰 코멘트별 대응 (하나하나)

**이 실험이 정조준하는 코멘트:**
- **Meta(AC)**: *"differentiation from relative-representation methods"* + *"controlled baselines
  isolating the benefits of **anchors**"* → fixed(ASIF) vs learned(PAL) 통제 비교가 정확히 그것.
- **VQng W1**: *"novelty = replacement of fixed anchors with learnable ones … what is technically
  new?"* → 치환의 가치를 정량화(learnable이 +X). "cosmetic 조립이 아니라 성능을 만드는 핵심".
- **9q1o Q3**: *"how much gain from **anchor formulation** vs CAP vs token-level?"* → 앵커(학습)
  갈래 담당.
- **9q1o Weakness(controlled baselines)** + *"prototype-based alignment와 구분"* → fixed-kmeans(=
  prototype) vs PAL이 직접 답(§5).

**이 실험이 *안* 다루는 것(정직하게 — 다른 실험 소관):**
- VQng W2/Q1(CAP vs mean/std-attn) → **CAP-isolation**(완료). Q2(CAP 이름) → 문서.
- VQng Q3, 9q1o Q4(projection-free 실익/저데이터 overfit/data regime) → **data-scaling**(진행 중).
- 9q1o Q2, VQng W3(fine-grained 직접 증거) → grounding/seg. (단 이 실험의 seg 결과가 부수 지원.)
- 2Mov 전반(contrastive baseline/[1,2,3]/CLIP-BLIP/도메인) → 각각 별도. (2Mov W2의 [2]Chen
  relative-rep는 이 실험이 positioning 논거 제공.)

---

## 7. 그룹핑 — 한 실험이 묶어 답하는 코멘트 + Q3 3분해

**fixed-vs-learnable 실험 1개** → **Meta + VQng-W1 + 9q1o-Q3(앵커) + 9q1o-controlled + prototype**
(전부 "앵커 *학습*의 기여 격리·정량화").

**9q1o-Q3의 3갈래는 3실험을 합쳐야 완성** (rebuttal 킬러 표):
| Q3 갈래 | 담당 실험 | 상태 |
|---|---|---|
| token-level features | token-level linear/mlp (λ0, pin) | ✅ 완료 |
| CAP (pooling) | CAP-isolation (mean/MAP/CAP) | ✅ 완료 |
| **anchor formulation (learning)** | **이 fixed-vs-learnable 실험** | ⬜ GPU1 |
→ 누적 분해표(`token-only → +CAP → +fixed anchor → +learned anchor = PAL`)로 제시.

---

## 8. 실행 범위 (갱신 — 구현은 v1 참조, 이미 구현됨)

**메인(고정 데이터, pooled 레벨) — 앵커 구성 사다리** = 축 B 비교:
- `fixed-random`(floor, 붕괴) → `fixed-ASIF`(pooled, topk/p 강화) → `fixed-kmeans`(페어 보존)
  → `PAL-CLS`(learned/pooled) → **PAL-CAP(헤드라인)**.
- `fixed_cap`은 near-chance라 "**CAP은 학습앵커 의존**" ablation으로만 리포트.

**스케일링(축 A, 의미 있는 것만)** — 2Mov-W3:
- **PAL(K=512)** across n (보유) vs **ASIF(K=n)** across n. fixed-K=512는 flat 기준선 1개만.

**강한 ASIF(strawman 방지)**: `topk`×`sim_exponent` 소규모 sweep(예: topk∈{128,256,none},
p∈{1,4,8})로 ASIF 최강 config 선택. **PAL은 topk=none/p=1(dense)로 잘 됨 → "ASIF는 손튜닝 정제
의존, PAL은 불필요"** 추가 논거.

**구현 상태**: 레이어(`fix_anchors`/`set_anchors_from_data`/`topk`/`sim_exponent`)·트레이너
(fit phase D: 쌍 앵커 주입+학습 스킵)·config(`configs/asif/vitl_roberta/{fixed_cap,fixed_pooled}`)
**모두 완료**. 신규 필요: (a) 강한 ASIF용 topk/p config 값, (b) fixed-random 위해 "set_anchors
스킵" 플래그(소), (c) fixed-kmeans 위해 쌍 클러스터링 앵커 주입(소), (d) ASIF K=n 스케일링 config.

---

## 9. 관련 문서
- **v1(구현)**: `docs/asif_fixed_anchor_baseline.ko.md` — 레이어/트레이너/ablation grid/코드
  체크리스트/Flavor A·B.
- CAP-isolation: `docs/cap_pooling_ablation.ko.md`. 데이터스케일링·의료: 각 doc.
- 원전: `ref_papers/ASIF.pdf`, `ASIF/`(공식 코드). Moschella et al. ICLR'23(상대표현 뿌리).
