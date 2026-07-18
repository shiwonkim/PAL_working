# ASIF-style Fixed-Anchor Baseline — 실험 설계 & 구현 방법

> 상태: 제안(draft). NeurIPS 2026 rebuttal 대비. PAL(learnable anchors)의 핵심
> 기여를 검증하기 위해, 같은 태스크에서 **fixed anchors**를 쓰는 선행연구 ASIF와
> 통제 비교한다. 참고 구현: `ASIF/`(공식 레포, 로컬 전용 클론 — `.git/info/exclude`),
> 논문 `ref_papers/ASIF.pdf`.

## 1. 동기 — 왜 이 비교가 (거의) 필수인가

PAL은 relative-representation 계보의 연장이다:

- Moschella et al. (ICLR'23) — 상대표현으로 latent space 소통, **fixed anchors**.
- **ASIF** (Norelli et al., NeurIPS'23) — 두 frozen 단일모달 인코더를 **ground-truth
  이미지-텍스트 pair 집합에 대한 유사도**로 정렬. **학습 없음(non-parametric)**,
  fixed anchors, sparse(top-k) + exponent.
- **PAL** (본 연구) — **K개 learnable anchors + CAP**, projection-free, 학습 기반.

따라서 PAL의 novelty를 한 문장으로 요약하면 이 계보의 **"learning" 스텝**이다.
relative-representation 문헌을 아는 리뷰어는 반드시 *"anchor를 학습하는 게 fixed
anchor 대비 무엇을 주느냐?"* 를 묻는다. 이 실험이 없으면 PAL의 핵심 기여(=projection-free
**anchor learning**)가 검증되지 않은 채 남는다.

## 2. 핵심 차이 (설계의 축)

ASIF와 PAL의 결정적 차이는 **anchor 간 modality 대응을 어디서 얻는가**이다.

- **ASIF**: anchor `k`가 실제 (이미지, 캡션) pair다. 이미지는 image-anchor
  `A_img ∈ R^{K×D_img}`에, 텍스트는 text-anchor `A_txt ∈ R^{K×D_txt}`에 대해 유사도를
  재되, **같은 index k로 비교**한다. "이미지-anchor-k ↔ 텍스트-anchor-k" 대응은
  **데이터쌍에서 공짜로 주어진다.** (`ASIF/relreps.py:relative_represent` — `sim(y, basis)`의
  top-k, 그 뒤 exponent `p`; `A_img`/`A_txt`가 같은 pair 인덱스.)
- **PAL**: image-layer와 text-layer가 각자 자신의 anchors `A_img`, `A_txt`를 가지며,
  "image-anchor-k ↔ text-anchor-k" 대응이 **contrastive 학습으로 만들어진다**
  (`src/models/alignment/pal.py`, profile `(B,K)` 둘을 CLIP loss로 정렬).

→ **PAL은 대응을 학습하고, ASIF는 데이터쌍에서 얻는다.** 이 실험은 정확히 그 "학습"의
가치를 분리·정량화한다.

부차적 차이(통제해야 함): ASIF는 **pooled 임베딩 + sparse top-k + exponent**, PAL-token은
**CAP 토큰 pooling**. 이 둘을 섞어서 비교하면 "토큰 덕분에 이겼다"는 반박을 부른다 → §4의
ablation grid로 분리한다.

## 3. 가설

- **H1**: learnable anchors는 같은 K에서 fixed(data-pair) anchors보다 정렬 품질이 높다
  (retrieval/zero-shot).
- **H2 (learning이 이기는 축)**: 격차는 (i) **적은 K**, (ii) **인코더-family 교체**,
  (iii) **적은 학습쌍** 조건에서 커진다. 단일 정확도보다 이 축들에서의 우위가 임팩트 있다.
- **H3 (공정성)**: PAL의 우위는 CAP(토큰)만이 아니라 **anchor 학습 자체**에서 온다
  (fixed+CAP < learned+CAP를 보여 분리).

## 4. 실험 설계 — Ablation Grid

인코더/데이터/K/평가를 **완전히 동일**하게 두고 두 축(anchor, pooling)만 바꾼다.

| variant | anchors | pooling | 코드상 실현 | 역할 |
|---|---|---|---|---|
| **ASIF-faithful** | fixed (data pair) | pooled + sparse top-k + exp `p` | Flavor B (§5.2, ASIF 레포 재사용) | 진짜 선행연구 baseline |
| **fixed / pooled** | fixed (data pair) | pooled (CLS) | Flavor A + CLS | 통제된 fixed-anchor |
| **fixed / CAP** | fixed (data pair) | CAP | Flavor A + token | CAP만의 기여 분리 |
| **learned / pooled** | learned | pooled (CLS) | 기존 PAL-CLS 설정 | 학습만의 기여 분리 |
| **learned / CAP (=PAL)** | learned | CAP | 기존 token_k512 | 제안 |

- **fixed/pooled → learned/pooled**: 같은 pooling에서 학습의 효과(H1) = 가장 깨끗한 셀 비교.
- **fixed/CAP → learned/CAP**: CAP를 고정한 채 학습의 효과(H3).
- **ASIF-faithful**: ASIF 자기 최적 config(큰 dictionary, top-k, exponent)를 허용해
  "우리 fixed 버전은 진짜 ASIF가 아니다"는 반박을 차단.

우선순위(rebuttal): **fixed/pooled vs learned/pooled** (H1) → **fixed/CAP vs PAL** (H3) →
**ASIF-faithful** (선행연구 재현) → K-효율/인코더-교체 sweep (H2).

## 5. 구현 방법

### 5.1 Flavor A — "frozen-anchor" 변형 (통제 비교, 우선)

기존 PAL 파이프라인을 거의 그대로 쓰되 anchor를 **데이터쌍으로 초기화 + 동결**한다.
projection-free이므로 anchor를 얼리면 **학습 파라미터가 0개** → 학습 loop를 건너뛰고
바로 체크포인트만 저장, 이후 `src/eval.py`로 평가한다(PAL과 동일 경로).

구현은 **PAL 레이어에 플래그 추가**가 최소 침습적이다(새 클래스 대신).

**(a) `PALAlignmentLayer` 확장** (`src/models/alignment/pal.py`)
- 새 kwargs: `freeze_anchors: bool = False`, `anchor_init: "random" | "data" = "random"`,
  그리고 ASIF 충실도용 옵션 `topk: int | None = None`, `sim_exponent: float = 1.0`.
- `freeze_anchors=True`면 `self.anchors.requires_grad_(False)`.
- `anchor_init="data"`면 anchors는 **밖에서 주입**한다(빌드 시점엔 데이터가 없으므로).
  주입 API 하나 추가: `set_anchors_from_data(vecs: Tensor)` — `vecs (K, D)`를
  `F.normalize` 후 `self.anchors.data`에 복사하고 freeze.
- forward에 sparsify/exponent 훅(옵션): profile 계산 뒤 `topk`가 있으면 top-k 외 0,
  `sim_exponent != 1`이면 유사도를 `p`제곱 (ASIF ii). **기본값은 no-op**이라 기존 PAL 불변.

**(b) 데이터쌍 anchor 주입** (`src/training/trainers/alignment_trainer.py`,
`_train_layer_pair` 빌드 직후 = 현재 `line 1333–1350`)
- 빌드 직후, `config["training"].get("fixed_anchors")`가 참이면:
  1. `K = alignment_layer_kwargs["num_anchors"]`.
  2. 준비된 train 텐서에서 **같은 행 인덱스** `idx`를 `random_state`로 재현가능하게 K개 샘플
     (`layer_image_features_train`, `layer_text_features_train`은 이미 정렬된 pair — dedup 후
     행 대응 유지). 토큰 모드면 pooled 벡터가 필요하므로 anchor용으로 **CLS/mean-pool**해서
     `(K, D)` 벡터를 만든다(테스트 토큰은 CAP로 이 고정 anchor에 pool).
  3. `alignment_image.set_anchors_from_data(img_anchor_vecs)`,
     `alignment_text.set_anchors_from_data(txt_anchor_vecs)`.
  4. 재현/해석성 위해 `idx`를 체크포인트 메타에 저장(ASIF의 interpretability 대응).

**(c) 학습 스킵 경로** (`_train_layer_pair`)
- `fixed_anchors`면 LR finder/epoch loop를 건너뛴다. 파라미터가 0개라 optimizer 구성이
  비므로 **명시적 분기**가 안전하다: 빌드+anchor주입 → `save_checkpoint`(best=현재) →
  return. (한 번의 `validate()`로 val loss만 로깅해도 좋음.)
- 체크포인트는 기존 `serialize_alignment_layer` 포맷 그대로 → `eval.py`가 수정 없이 로드.

**(d) config 예시** (`configs/asif/vitl_roberta/fixed_pooled.yaml` 등)
```yaml
defaults: !include ../../default.yaml
overrides:
  alignment: { llm_model_name: ".../all-roberta-large-v1",
               lvm_model_name: "vit_large_patch14_dinov2.lvd142m" }
  features: { dataset: coco, layer_img: 23, layer_txt: 24 }
  training:
    token_level: false           # fixed/pooled 셀 (CAP 셀은 true)
    fixed_anchors: true          # ← 새 플래그: 데이터쌍 anchor + no-train
    alignment_layer_name: PALAlignmentLayer
    alignment_layer_kwargs:
      num_anchors: 512           # PAL과 동일 K (apples-to-apples)
      freeze_anchors: true
      anchor_init: data
      pool_temperature: 0.03     # CAP 셀에서만 의미
      # topk: 64                 # (옵션) ASIF 충실도
      # sim_exponent: 4.0        # (옵션) ASIF 충실도
```
평가는 기존과 동일: `python -m src.eval --config_path <cfg> --ckpt <...> --zs ... --rt ...`.
(B,K) profile을 내므로 retrieval/zero-shot 루프 **무수정**.

### 5.2 Flavor B — Faithful ASIF (선행연구 재현, standalone)

ASIF 공식 레시피를 그대로 재현해 "진짜 ASIF" 수치를 만든다. 학습·트레이너를 안 거치고
캐시된 임베딩 위에서 직접 계산 → `ASIF/relreps.py` 로직 재사용.

- 입력: 우리 파이프라인이 이미 만든 **pooled(cls/avg) 임베딩 캐시**(image/text, train=anchor
  풀, eval=test). `results/features/*-cls.npy` / `*-avg.npy` 재사용.
- 절차(ASIF §2):
  1. anchor 풀에서 K(또는 큰 dictionary n) pair 샘플 → `A_img (n,D_img)`, `A_txt (n,D_txt)`.
  2. 테스트 이미지/텍스트를 각각 `A_img`/`A_txt`에 대해 relative-represent:
     `relative_represent(z, basis, non_zeros=k)` (top-k) → sparse.
  3. exponent `p` 적용(`relreps.py`의 값 거듭제곱), row-normalize(`normalize_sparse`).
  4. retrieval/zero-shot: image relrep vs text relrep을 **같은 anchor 인덱스**로 비교
     (`zero_shot_classification` 참고).
- 스크립트: `src/evaluation/asif_eval.py`(신규, 얇은 래퍼) — 우리 데이터 로더 +
  `ASIF/relreps.py` 함수. 하이퍼파라미터 `(n, k, p)`는 ASIF처럼 소검증셋에서 튜닝.
- 리포트: 표에 **ASIF (faithful)** 행으로. 우리 인코더/데이터셋에 맞춘 값이라 통제 비교와
  선행연구 재현을 동시에 만족.

## 6. 평가 / 지표 / 데이터

- **인코더/데이터 고정**: DINOv2 ViT-L + RoBERTa-large, COCO train, layer (23,24) —
  현재 seed 실험과 동일 축(공정성).
- **지표**: retrieval(flickr30, coco_karpathy) + zero-shot(기존 suite: stl10/cifar100/
  dtd/eurosat/caltech101 등). 전부 기존 `src/eval.py` 재사용(Flavor A) / `asif_eval.py`(Flavor B).
- **도메인 보존**(리뷰 대비): 새 실험 없이 기존 zero-shot suite에 EuroSAT(위성)/DTD(텍스처)/
  GTSRB(표지판) 등 **도메인이 다른 벤치가 이미 포함** → "PAL이 도메인 시프트에서 fixed보다
  보존이 낫다"를 **재해석**으로 주장.

## 7. 예상 결과 해석표

| 관찰 | 결론 |
|---|---|
| learned/pooled > fixed/pooled (같은 K) | anchor 학습의 순효과 (H1) ✅ |
| learned/CAP > fixed/CAP | 우위가 CAP만이 아님 (H3) ✅ |
| 격차가 작은 K에서 확대 | learnable anchor의 표현 효율 (H2) |
| ASIF-faithful ≈ fixed/pooled | 우리 fixed 버전이 진짜 ASIF임을 확인(공정) |
| PAL ≳ ASIF-faithful | 선행연구 대비 최종 우위 (헤드라인) |

부분 결과도 논문거리: fixed가 특정 셋에서 경쟁력 있으면 *"학습은 품질 X를 얻고 fixed는
편집성/무학습 Y를 얻는다"*는 trade-off로 정직하게 프레이밍.

## 8. 코드 변경 요약 (구현 체크리스트)

- [ ] `src/models/alignment/pal.py`: `freeze_anchors`/`anchor_init`/`topk`/`sim_exponent`
      kwargs + `set_anchors_from_data()` + forward의 sparsify/exponent 훅(기본 no-op).
- [ ] `src/training/trainers/alignment_trainer.py` `_train_layer_pair`: `fixed_anchors`면
      빌드 직후 데이터쌍 anchor 주입 + 학습 loop 스킵 + 체크포인트 저장(+idx 메타).
- [ ] `configs/asif/vitl_roberta/{fixed_pooled,fixed_cap}.yaml` (+ 인코더-family 변형).
- [ ] (Flavor B) `src/evaluation/asif_eval.py`: `ASIF/relreps.py` 재사용 standalone 평가.
- [ ] **loadability 검증**(CLAUDE.md 규칙): fixed-anchor 체크포인트 저장→로드→forward 일치 확인.

## 9. 리스크 / 주의

- **공정성**: K, 인코더, 데이터셋, eval 프로토콜을 PAL과 **완전 동일**하게. 토큰 vs pooled를
  grid로 분리(안 하면 반박 유발).
- **파라미터 0개 경로**: optimizer/LR finder가 빈 파라미터에서 죽지 않도록 학습 스킵 분기 필수.
- **anchor 샘플 분산**: fixed anchor는 어떤 K개를 뽑느냐에 민감할 수 있음 → 여러 seed로 anchor
  샘플을 바꿔 평균±표준편차 리포트(ASIF의 anchor 선택 민감도와 동일 이슈).
- **ASIF 충실도**: sparse top-k + exponent는 ASIF 성능에 중요(논문 §2 i,ii) → Flavor B에선
  반드시 포함, 소검증셋 튜닝.

## 10. 관련 자료

- `ref_papers/ASIF.pdf`, 공식 코드 `ASIF/`(relreps.py, relrepsutils.py, embdatasets.py).
- Moschella et al., *Relative representations enable zero-shot latent space communication*,
  ICLR 2023 (상대표현 fixed-anchor의 뿌리).
- 본 repo: `src/models/alignment/pal.py`(learnable anchor+CAP),
  `docs/pal_generative_vlm_plan.ko.md`(상대표현의 또 다른 확장 축).
