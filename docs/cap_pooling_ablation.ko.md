# CAP pooling ablation — 방어 논리 (rebuttal 노트)

리뷰어들이 "CAP의 이득이 CAP **특유의 설계** 때문이냐, 아니면 **아무 weighted token
aggregation**(mean, generic attention)이면 되냐"를 물었다. 이 문서는 그 격리 실험의 설계와
방어 논리를 정리한다. (실험 자산: `PAL-mean`(완료), `linear/mlp-token`(완료),
`AttnPoolAlignmentLayer`(신규, single-query MAP), `PAL-CAP`(메인).)

## 1. 리뷰어가 요구한 것

- **VQng W2**: "CAP은 anchor-wise softmax pooling인데 **standard slot attention / soft
  assignment와 구조적으로 유사**하다. CAP이 이들과 어떻게 다른지/나은지 논의가 부족하다.
  **대안 pooling과의 비교/ablation** (mean pooling, **standard cross-attention with learned
  queries**)이 필요하다."
- **VQng Q1**: "Table 5에서 CAP 추가 시 retrieval **+10.9**, seg **+7.6**. 이게 CAP의
  **anchor-wise softmax 구조 특유**인지, **아무 weighted aggregation**이면 되는지 불명확.
  CAP을 (a) uniform mean pooling, (b) **standard cross-attention with learned query vectors**와
  비교했나?"
- **Meta review**: "anchors와 CAP의 이득을 token-level features / **generic weighted pooling**
  으로부터 격리하는 **controlled baseline 부재**."
- **9q1o Q3**: "이득이 anchor formulation + CAP 때문인지, **general token-level
  representation** 때문인지 얼마나?"

## 2. CAP이란 (formal, `src/models/alignment/pal.py`)

토큰 `z ∈ (B,T,D)`, 학습 anchor `A ∈ (K,D)`:

    sim   = normalize(z) @ normalize(A)ᵀ        # (B,T,K)  코사인 유사도
    attn  = softmax(sim / τ,  dim=tokens)       # (B,T,K)  **토큰 축** softmax
    profile = Σ_t (attn ⊙ sim)                  # (B,K)    K차원 상대표현
    output  = L2normalize(profile)

핵심 성질: **projection-free**(W 없음, 코사인만) · 출력이 **K차원 유사도 profile(스칼라)** ·
softmax가 **토큰 축**(각 anchor가 독립적으로 토큰을 pooling, anchor끼리 경쟁 없음).

## 3. CAP ≠ Slot Attention (VQng W2 직접 반박)

Slot Attention (Locatello et al., NeurIPS'20)은 K개 slot이 입력을 두고 **경쟁**하는
object-centric 메커니즘이다. CAP과 핵심에서 다르다:

| | **CAP** | **Slot Attention** |
|---|---|---|
| softmax 방향 | **토큰 축** (anchor 독립, 경쟁 X) | **slot 축** (slot끼리 입력 경쟁) |
| 반복 | 없음 (1-pass) | 반복 + **GRU** 업데이트 |
| projection | **없음** (코사인, projection-free) | 있음 (W_q/W_k/W_v) |
| 출력 | K차원 **유사도 profile (스칼라)** | K개 **feature 벡터** |

→ CAP은 slot attention의 "경쟁적 soft-assignment"가 **아니다**. softmax 방향이 반대이고,
반복/GRU가 없고, projection-free이며, 출력이 relative-representation이다. "유사하다"는 지적은
표면적 유사성(둘 다 K개 학습 벡터 + soft aggregation)에 근거하며, 위 4가지가 실질적 차별점이다.

## 4. Controlled ablation — 2×2 요인 격자

CAP은 두 가지를 **동시에** 한다: (1) **표현**(anchor 상대표현 vs projection 절대표현),
(2) **pooling**(anchor-softmax vs mean vs learned-attention). 두 축을 각각 고정하며 격리한다:

| 표현 \ pooling | **mean** | **learned-attn (MAP)** | **anchor-softmax (CAP)** |
|---|---|---|---|
| **절대(projection)** | linear-token / mlp-token | **AttnPool (single-q MAP)** | — |
| **상대(anchor)** | **PAL-mean** | — | **PAL-CAP** |

- **표현 효과** (pooling=mean 고정): `linear-token(proj+mean)` → `PAL-mean(anchor+mean)`
  = anchor 상대표현의 순수 기여.
- **pooling 효과** (표현=anchor 고정): `PAL-mean` → `PAL-CAP` = CAP의 순수 기여.
- **generic attention pooling 대조**: `AttnPool(MAP)` vs `PAL-CAP` = "학습된 attention
  pooling(표준)이면 되냐, 아니면 CAP의 코사인-anchor-softmax + projection-free가 필요하냐".

모든 셀은 동일 세팅(vitl_roberta, layer (23,24), d=512, token-level, **structure_lambda=0**,
seed42)이라 직접 비교 가능하다. (structure_reg를 뺀 이유: 모든 token 방법(PAL/fa/sail)이 λ0이며,
structure_reg는 pooled(2D) 전용 규제 — `docs`/코드 참조.)

## 5. Single-query MAP (`AttnPoolAlignmentLayer`)

리뷰어의 "standard cross-attention with learned query"에 정확히 대응하는 **표준** 헤드 =
Multihead Attention Pooling(ViT/SigLIP/CoCa). 1개 학습 query가 토큰에 cross-attend하여 d차원
벡터로 pooling:

    kv     = W_in(z)                              # 토큰 projection (B,T,d)
    pooled = MHA(query, kv, kv)                   # W_q/W_k/W_v projection + softmax(토큰축)
    out    = L2normalize(pooled + MLP(LN(pooled)))# residual MLP head

CAP과의 대조:
- **projection 있음** (W_in, MHA의 W_q/W_k/W_v, MLP head) ↔ CAP은 projection-free
- **학습된 attention** ↔ CAP은 고정 코사인 유사도
- **feature pooling → d차원 벡터** ↔ CAP은 유사도 → K차원 상대표현
- **파라미터**: MAP **≈3.68M** vs CAP **≈0.5M** (K anchor만). MAP이 ~7× 무겁다.

→ MAP(projection-heavy)이 CAP(projection-free)을 **못 이기면**, 특히 **저데이터**에서 MAP이
더 오버피팅하여 격차가 벌어지면: (i) CAP의 이득은 "아무 attention pooling"이 아니라 그 특유
설계 때문(우려 A), (ii) projection 제거가 저데이터 일반화에 유리(우려 B, projection-free)임을
동시에 입증.

## 6. "1개 query 말고 K개로 맞춰야 하는 것 아니냐" 선제 방어

VQng이 "query **vectors**"(복수)라 했으므로 "CAP의 K anchor에 맞춰 K개 query를 쓰라"는 반론이
가능하다. 방어:

1. **역할이 다르다.** CAP의 K anchor는 **출력 basis**(출력이 K차원 상대표현)이지 pooling
   용량이 아니다. MAP의 query는 **pooling 가중치 생성기**이고 출력은 d차원. 공정성 기준은
   "query 수"가 아니라 **출력 차원(둘 다 512) + token-level attention pooler**라는 점이다.
2. **CAP의 (B,K) 유사도-profile 출력은 설계의 일부**다. 표준 K-query attention pooler는
   feature를 pooling해 `(B,K,d)`를 내며, 이를 `(B,K)` 유사도-profile로 바꾸는 것은 자연스럽지
   않다 — 즉 "K-query로 맞춘 CAP"은 **빠진 baseline이 아니라 CAP 고유 설계와의 차이 그 자체**.
3. **(옵션) 완전 방탄용 secondary**: K-query Perceiver/Q-Former식 pooler(K=512 query, 표준
   QKV attention → (B,K,d) → 축약)를 추가하면 "같은 K, 코사인-anchor(CAP) vs 학습-QKV-attention"
   대조가 되어 이 반론을 완전 차단. 단 출력 축약이 비표준이라 복잡 — 리뷰어가 더 밀 때만.

## 7. 이 실험이 답하는 리뷰 우려

- **A (CAP/anchor 격리)**: 2×2 격자 + MAP 대조 → "token 정보/generic pooling이 아니라
  anchor+CAP 설계 때문" (Meta, VQng W2/Q1, 9q1o Q3).
- **B (projection-free 이점)**: projection-heavy MAP/linear/mlp vs projection-free CAP,
  저데이터서 격차 확대 → "projection 제거가 일반화에 유리" (VQng Q3, 9q1o).
- **C (저데이터)**: 데이터 크기별로 격자를 그리면 CAP 우위가 scarce에서 최대임을 정량화
  (Meta, 9q1o, 2Mov W3) — [[cap-ablation-across-scales]] (선택).

## 참고 (구현/실행)

- 레이어: `src/models/alignment/attn_pool.py` (`AttnPoolAlignmentLayer`), factory 자동 등록.
- config: `configs/attn_pool/vitl_roberta/attn_pool_d512_token.yaml` (λ0, 핀 23/24, seed42).
- 실행: `WANDB_NAME=attn_pool_d512_token_seed42 ... python -m src.train
  --config_path configs/attn_pool/vitl_roberta/attn_pool_d512_token.yaml --seed 42`,
  이후 eval(zs/rt/seg)로 격자에 삽입.
- 스모크: forward(token/CLS/masked) + 체크포인트 라운드트립(Δ=0) 검증 완료.
