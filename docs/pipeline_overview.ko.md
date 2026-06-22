# 파이프라인 개요 — 학습 & 추론 아키텍처

> 리팩터링 전/중에 코드베이스를 이해하기 위한 레퍼런스.
> **file:line 참조는 2026-06 시점 기준(커밋 ~2ae0859)**이며 코드가 바뀌면
> 어긋납니다 — *구조/개념*을 신뢰하고 정확한 줄은 다시 grep 하세요.
> (영어 원본: `docs/pipeline_overview.md`)

---

## 0. 핵심 설계 철학

**인코더(ViT / RoBERTa)는 학습 루프에서 절대 안 돕니다.** 모든 feature를 미리
추출해 디스크에 캐시하고, 그 캐시 위에서 **alignment layer만** 학습합니다.

이 한 가지가 거의 모든 설계 선택을 설명합니다:
- epoch마다 인코더 forward가 없음 → 매우 빠름 → 1000 epoch 디폴트가 가능.
- feature 텐서 전체를 메모리에 들고 배치 인덱싱 → 토큰 캐시가 크면 RAM 압박
  (LAION 메모리 이슈의 근원 — `docs/laion_reimplementation_TODO.md` 참고).
- alignment layer는 **modality마다 따로** 생성 (같은 클래스, 두 인스턴스):
  `alignment_image`, `alignment_text`. config의
  `training.alignment_layer_name`으로 `AlignmentFactory`가 생성
  (`src/trainers/alignment_trainer.py` ~1839).
- **CLS vs Token** 이라는 하나의 축이 추출 → 학습 → 모든 평가 경로를 관통 (§5).

---

## 1. 학습 파이프라인

진입: `python src/train_alignment.py --config_path X.yaml`

```
src/train_alignment.py
  ① config 로드 + merge(defaults, overrides)                   ~103
  ② load_dataset(): coco/coco2017/flickr30 = caption 데이터셋, ~18-79
     나머지는 ImageTextDataset로 래핑
  ③ eval 데이터셋 로드 (zero_shot / retrieval)                 ~176-194
  ④ dispatch:                                                  ~205-210
       training.cca=true  → CSATrainer (closed-form CCA)
       training.clip=true → CLIPEvalTrainer
       else               → AlignmentTrainer  ← 메인 경로
```

### AlignmentTrainer.fit()  (`src/trainers/alignment_trainer.py` ~1181-2061)

**(a) Feature 추출 & 캐싱** — `get_image_features` (~282-425) /
`get_text_features` (~172-280)
- pool 모드가 모양을 결정:
  - `pool_*="cls"` → CLS 토큰 → `(N, D)`          (CLS 메소드)
  - `pool_*="avg"` → 토큰 평균 → `(N, D)`
  - `pool_*="none"` + `layer_*` → 단일 레이어 토큰 → `(N, T, D)`  (token 메소드)
- 캐시 경로 `results/features/{model}-{dataset}-{suffix}.npy`; `mmap=True`로 재로드.
- **이미지 dedup at extraction** (~322-358): COCO는 캡션당 행이 있어 이미지가 ~5×
  중복 → 추출 시 유니크 이미지 + 인덱스 맵만 (디스크/시간 5× 절약).

**(b) Layer Selection** — `compute_layer_alignment` (~427-570)
- config에 `layer_img/txt`가 없을 때만 실행. **mutual-kNN**으로 모든
  `(img_layer × txt_layer)` 쌍을 점수화, `best_only`로 top 쌍 선택.
  COCO2014: ViT-L→(23,24), ViT-S→(11,6), ViT-B→(11,12).

**(c) fit 루프** (~1499-2061), 선택된 레이어 쌍마다:
- feature 슬라이스; token_level이면 `_load_token_features_for_layer` (~710-814)가
  `(N,T,D)` + 텍스트 mask `(N,T)` 로드.
- factory로 alignment layer 생성; loss 생성; (선택) LR finder
  (`base_trainer.py` ~147-314); epoch 루프.
- **한 배치** (`train()`, ~2065-2349):
  ```
  image_feats = image_features[i:end].to(device)          # 캐시 인덱싱
  aligned_img = alignment_image(image_feats[, cls_attn])  # (B,D|T,D) → (B,K)
  aligned_txt = alignment_text(text_feats[, mask])        # (B,D|T,D) → (B,K)
  loss = CLIPLoss(aligned_img, aligned_txt, ...)
  loss.backward(); clip_grad; opt.step(); sched.step()
  ```
- early stopping; best val일 때 `save_checkpoint`가 `alignment_image`,
  `alignment_text`, optimizer, **config 통째**를 저장 (~2014).

**알아둘 token 디테일:** token_level일 때 feature가 처음엔 CLS stub `(N,1)`이었다가
학습 시점에 실제 `(N,T,D)`로 교체됨 → 그래서 `image_dim = features.shape[-1]`
재계산(~1818)이 필요. 없으면 `input_dim`이 1로 잡힘.

### CSATrainer (`src/trainers/csa_trainer.py` ~80-226)
Closed-form CCA (`NormalizedCCA`, `src/alignment/cca_class.py`); token-level 없음,
gradient 학습 없음 — canonical variate를 해석적으로 풀고 바로 평가.

---

## 2. Alignment Layer Zoo

전부 `BaseAlignmentLayer(input_dim)` 상속 (`src/alignment/base_alignment_layer.py`),
`forward(z, mask?)` 구현, `@AlignmentFactory.register()`로 등록
(`alignment_factory.py` + `__init__.py`의 `initialize_package_factory` 자동 import,
`src/utils/base_factory.py`). 출력은 항상 **`(B, K)` L2-정규화 profile** → CLIPLoss로.
트레이너가 modality마다 하나씩 만들고 `set_modality(...)` 있으면 호출 (~1839-1856).

| 메소드 | 클래스 (파일) | forward 요지 |
|---|---|---|
| Linear | `LinearAlignmentLayer` (`linear_alignment_layer.py`) | `Linear(z)`; 3D → masked mean-pool; 선택 L2 |
| MLP | `MLPAlignmentLayer` (`mlp_alignment_layer.py`) | 쌓은 ReLU MLP |
| STRUCTURE "MLP" | `ResLowRankHead` (`mlp_alignment_layer.py` ~51) | skip `P(z)` + 게이트된 low-rank residual `α·W₂(GELU(W₁z))`, `α=σ(logit)` 학습 |
| **PAL-CLS** | `PALAlignmentLayer` (`pal.py` ~57) | `normalize(normalize(z) @ normalize(anchors)ᵀ)` → `(B,K)` 코사인 profile |
| **PAL-Token ★** | `PALTokenAlignmentLayer` (`pal_token.py` ~116) | **CAP**, 아래 참고 |
| FreezeAlign | `FreezeAlignAlignmentLayer` (`freeze_align.py` ~195) | `set_modality`; img = patch-mean + CLS proj; txt = masked-mean + MLP |
| SAIL | `SAILStarMLP` (`sail_star_mlp.py` ~81) | `set_modality`; GLU `g(ReLU6(f₁z)⊙f₂z)`, concat[cls,patch] |
| CSA | `cca_class.py` | closed-form CCA; nn 레이어 아님 / factory 미등록 |

### ★ CAP — Cross-Attention Pooling (`pal_token.py` ~116-184)
K개 학습 anchor `(K,D)`가 **곧 alignment 파라미터** (`projector_dim=0`, 다운스트림
MLP 없음). 토큰 입력 `z:(B,T,D)`에 대해:
```
ẑ = normalize(z, -1)                  # (B,T,D)
â = normalize(anchors, -1)            # (K,D)
sim = ẑ @ âᵀ                          # (B,T,K)  토큰×anchor 코사인
logits = sim / pool_temperature       # τ=0.03 디폴트
[ + betaₖ · log(cls_attn) ]           # 선택적 cls_attn_prior, anchor별 bias
logits = logits.masked_fill(~mask, -inf)
attn = softmax(logits, dim=1)         # (B,T,K)  anchor마다 토큰을 soft 선택
profile = (attn * sim).sum(dim=1)     # (B,K)    anchor별 가중합
return normalize(profile, -1)
```
직관: **각 anchor가 자기와 맞는 토큰에 softmax로 주목해 그 유사도를 모음.**
2D(CLS) 입력이면 PAL-CLS 경로로 fallback (~124-128). 이게 리팩터 대상 레이어이고,
cls_attn-prior 분기가 forward에 인라인돼 있음.

---

## 3. Loss (`src/loss/clip_loss.py`, `siglip_loss.py`)
```
logits = aligned_img @ aligned_txtᵀ / temperature
loss = (CE(logits, arange) + CE(logitsᵀ, arange)) / 2          # 양방향 InfoNCE
     + structure_lambda · structure_reg(...)                   # 선택적
```
- **STRUCTURE 정규화** (`clip_loss.py` ~57-152): 원본 임베딩의 유사도 구조를 정렬
  후에도 보존하도록 JS-divergence 페널티 (3D면 mean-pool 먼저).
  `structure_lambda`(~10), `structure_levels`, warmup.
- **SigLipLoss** 대안: 각 쌍 독립 sigmoid + 학습형 `logit_scale`/`logit_bias`.

---

## 4. 추론 / 평가 (모두 학습된 alignment_image/text를 그대로 적용)

공통: ckpt 로드 → `alignment_{image,text}.eval()` → 있으면 `set_modality`.

**(a) Zero-shot 분류** — `zero_shot_classifier.py:build_zero_shot_classifier`
(~34) + `AlignmentTrainer.evaluate_zero_shot_classification` (~2516)
- 클래스명 × 템플릿 → 텍스트 임베딩 → `alignment_text` → 템플릿 평균 →
  `(num_classes, K)` 분류기; 이미지 → `alignment_image` → `(B,K)`;
  `100·img @ 분류기ᵀ` → argmax.
- **`token_level_zero_shot`가 결정적**: true면 클래스 템플릿을 **CAP/token 경로**
  (`pool_txt="none"` + mask)로, false면 CLS 경로로 만듦. 틀리면 분포 불일치로
  점수가 랜덤처럼 나옴. token 메소드는 반드시 true.

**(b) Retrieval** — `retrieval.py:retrieval_metrics_df` (~29) +
`evaluate_retrieval` (~2854)
- feature → alignment → `(N,K)` → 정규화 → `img @ txtᵀ` → topk →
  I2T/T2I R@1/5/10, MAP@k. 다중 캡션 GT는 `image_name`으로 그룹.

**(c) Segmentation** — `zero_shot_segmentation.py` (CLI `main` ~1329)
- `SegmentationMethod` 서브클래스: `direct_cosine`, **`anchor_codebook`
  (PALAnchorCodebookMethod)**, `freezealign`, `linear_perpatch`.
- **factorized 디코딩**: `S_pa (P,K) @ S_ac (K,C) = (P,C)` — anchor를 명시적 bridge로;
  vs **direct**: 정규화 패치 @ 텍스트.
- 패치 유사도 → `√P×√P` 그리드 → bilinear 업샘플 → argmax → **mIoU-fg**(전경
  factorized). `auto_filter_methods`가 ckpt에 맞는 메소드만 실행.

**독립 진입점:**
- `rerun_eval.py` (~46): ckpt 재평가; ckpt 경로의 `(\d+, \d+)`로 layer 자동감지,
  `--token_level_zs` 오버라이드 (양쪽 서버 병합본).
- `zero_shot_segmentation.py:main`: 세그 CLI.

---

## 5. CLS ↔ Token 축 (전반 관통)

| 단계 | CLS (`token_level=false`) | Token (`token_level=true`) |
|---|---|---|
| 추출 pool | cls/avg → `(N,D)` | none → `(N,T,D)` + mask |
| alignment forward | `layer(z)` 2D | `layer(z, mask)` CAP |
| ZS 템플릿 | CLS 경로 | CAP 경로 (`token_level_zero_shot`) |
| 메소드 | Linear / MLP / CSA / PAL-CLS | PAL-Token / FreezeAlign |

같은 "if token: mask 경로; else: 2D 경로" 분기가 추출/fit/각 eval에 따로따로
재구현돼 있음 — 1순위 공통화 대상.

---

## 6. 리팩터링 관찰점 (시작점)

1. **`alignment_trainer.py`가 비대** (~2900줄; `fit()` 혼자 ~800): 추출 + layer
   selection + token/CLS 분기 + dedup + subsample + cls_attn + LR finder +
   train/validate + eval 3종이 한 클래스에. 책임 분리 대상.
2. **CLS/Token 분기 중복** — 추출/fit/zs/retrieval/seg에 흩어짐. 공통 추상화로.
3. **Feature I/O가 트레이너에 박힘** (캐시 로드/mmap/dedup이 fit 안). **FeatureStore**
   추상화로 빼면 정리도 되고 **LAION 메모리 재구현의 자연스러운 집**이 됨
   (`docs/laion_reimplementation_TODO.md`: virtual-concat + mmap + buffer-shuffle +
   prefetch).
4. **pool 모드/레이어 슬라이싱이 in-place `config[...]=` 오버라이드** 방식이 여러 곳
   → side-effect 위험. 명시적 인자 전달로.
5. **CAP 레이어**(`pal_token.py`)는 비교적 깔끔; cls_attn-prior 분기가
   forward에 인라인돼 있어 분리 가능.

---

## 빠른 파일 맵

| 영역 | 파일 |
|---|---|
| entry | `src/train_alignment.py` |
| trainers | `src/trainers/{alignment_trainer,base_trainer,csa_trainer,clip_eval_trainer}.py` |
| alignment 레이어 | `src/alignment/*.py` (+ `alignment_factory.py`, `base_alignment_layer.py`) |
| losses | `src/loss/{clip_loss,siglip_loss}.py` |
| eval | `src/evaluation/{zero_shot_classifier,retrieval,zero_shot_segmentation}.py` |
| re-eval CLI | `rerun_eval.py` |
| config | `configs/default.yaml` + `configs/<method>/<encoder>/*.yaml` |
