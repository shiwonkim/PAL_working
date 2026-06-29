# Continual Learning 벤치마크 구현 설계 (C-CLIP VLCL → PAL, A 경로)

> 작성: 2026-06-29. **계획 문서** — 코드 구현 전. 출처: C-CLIP (ICLR 2025,
> Liu et al.) 논문의 **벤치마크 세팅**만 차용 (논문의 방법론 LoRA+CKC는 구현 대상
> 아님 — PAL 기반 CL 메소드는 직접 제안 예정).

---

## 0. 목표 한 줄

C-CLIP의 **VLCL 벤치마크(8-task 멀티모달 CL + zero-shot 평가)** 를, **PAL 코드베이스
+ A 경로(인코더 frozen, anchor head만 task마다 학습)** 로 구현해서, 앞으로 "PAL을
continual하게 확장하는 메소드"를 개발/평가할 **실험 환경**을 깔아둔다.

---

## 1. C-CLIP VLCL 벤치마크 정의 (그대로 차용)

### 1.1 백본
- C-CLIP: **CLIP (ViT-B/16)** 사전학습 weight (ImageNet-1K zero-shot 67.73%).
- 우리도 **CLIP ViT-B/16로 맞춤** (사용자 결정). 단 A 경로라 **인코더는 frozen**,
  학습은 PAL anchor head만.

### 1.2 task 시퀀스 (8개, 각 데이터셋 = 1 task, domain-incremental, 추론 시 task-ID 불필요)
| # | 데이터셋 | 도메인 | 출처 |
|---|---|---|---|
| 1 | Flickr30K | 일반 실세계 | Plummer 2015 |
| 2 | COCO | 일반 실세계 | Chen 2015 |
| 3 | Pets | 반려동물 | Parkhi 2012 |
| 4 | Lexica | AI 생성 이미지 | Shen 2024 |
| 5 | Simpsons | 만화 | — |
| 6 | WikiArt | 미술 | Saleh & Elgammal 2015 |
| 7 | Kream | 의류 | — |
| 8 | Sketch | 스케치 | Chowdhury 2022 |

→ task 1→8 **순차 학습**. 각 task는 image-caption pair(멀티모달 retrieval 학습).

### 1.3 평가 3트랙
1. **Multimodal CL (핵심)** — 매 task 학습 후 **그때까지(또는 전체) task를 재평가**해서
   **I2T R@1, T2I R@1** retrieval + 평균. (Figure 5: task index 0~8 × 각 task 성능 곡선
   = forgetting 추적.)
2. **Zero-shot retrieval** — held-out **HAVG** (학습 안 함) I2T R@1.
3. **Zero-shot classification** — ImageNet, CIFAR-100, Flowers, DTD, Food101,
   StanfordCars 6개. **매 stage(task 0~8)마다 측정** + **PD**(Performance Degradation)
   = (원본 백본 정확도) − (최종 정확도). ↓ 좋음.

### 1.4 학습 설정 (C-CLIP 기준 — 참고값)
- 입력 224×224, **데이터셋당 40 epoch**, lr 1e-6 + 5-epoch warmup, batch 1024 (8×4090),
  symmetric cross-entropy(=CLIP loss). → 우리 PAL은 anchor만 학습하므로 epoch/lr/batch는
  PAL에 맞게 재튜닝 필요 (이 값은 "인코더 fine-tune" 기준이라 그대로 안 씀).

---

## 2. C-CLIP(B) vs 우리(A) — 핵심 차이와 함의 (반드시 인지)

| 항목 | C-CLIP (논문, B 경로) | 우리 (A 경로) |
|---|---|---|
| 학습 대상 | **CLIP 인코더에 LoRA** (인코더 fine-tune) | **인코더 frozen, PAL anchor head만** |
| feature 캐시 | 불가 (인코더 forward가 변함) | **가능** (인코더 고정 → FeatureStore per-task 캐시 재사용) |
| forgetting의 정체 | 인코더 지식 forgetting | **anchor head의 forgetting** |
| zero-shot 분류 PD | 인코더 zero-shot 능력 저하 | **anchor를 거친 zero-shot의 저하** |

**PD가 0이 아니다 (검증됨):** PAL의 zero-shot classification은 이미지·클래스텍스트를
**anchor head를 통과시킨 정렬 공간**에서 유사도를 잰다
(`evaluate_zero_shot_classification`: `aligned = alignment_image(image_feats)`;
`build_zero_shot_classifier`: `alignment_layer(...)`). → anchor가 task마다 바뀌면
zero-shot 성능도 바뀌므로 **PD ≠ 0**. 즉 A 경로여도 평가 3이 **"anchor의 zero-shot
forgetting"** 으로 살아있다. (이게 우리 메소드의 핵심 측정축.)

**즉 우리 벤치마크 = "frozen CLIP 인코더 위에서 PAL anchor를 task 시퀀스로 학습할 때의
망각/전이"를 재는 환경.** C-CLIP의 "인코더 forgetting" 축은 자연히 anchor 축으로 치환됨.

---

## 3. PAL 코드베이스 매핑 (무엇을 재사용 / 무엇을 신규)

| 벤치마크 요소 | 현재 코드 | 작업 |
|---|---|---|
| **CLIP 인코더** | `get_lvm`(timm ViT) + `get_llm`(HF AutoModel) — vision/text 별개 | **신규**: CLIP은 vision+text 한 모델이라 `get_lvm`/`get_llm`에 CLIP 경로 추가 (§4) |
| frozen + per-task feature 캐시 | `FeatureStore` (dataset별 캐시) | **재사용** — task = dataset이라 캐시가 곧 per-task |
| anchor head (PAL) | `PALAlignmentLayer` + `_train_layer_pair` | **재사용** + warm-start 옵션 추가 |
| 한 task 학습 | `fit()` (단일 데이터셋) | **재사용** (task마다 1회 호출) |
| **CL 오케스트레이터** | 없음 (train_subset이 "반복 fit" 선례) | **신규**: 8-task 시퀀스 루프 + anchor 체크포인트 이어받기 |
| retrieval/zero-shot 평가 | `evaluate_retrieval`, `evaluate_zero_shot_classification` | **재사용** |
| **forgetting 매트릭스** | 단일 task 평가만 | **신규**: stage t마다 task 1..t(또는 전체) 재평가 → 행렬 + 집계 |
| 8 task + HAVG + zs 6개 데이터셋 | coco/flickr만 있음 | **신규**: 나머지 로더 (일부 비표준) |

---

## 4. CLIP 인코더 통합 (가장 먼저 풀 기술 이슈)

PAL은 vision(timm)·text(HF)를 따로 뽑는데 CLIP은 한 모델 → 다음 결정 필요:
- **vision**: `timm`의 CLIP-ViT(`vit_base_patch16_clip_224.openai` 등) 또는 `open_clip`.
  `get_lvm`의 `return_nodes=blocks.{i}.add_1`(레이어별 토큰)이 CLIP-ViT 블록 구조와
  맞는지 확인. **어느 표현을 anchor 입력으로 쓸지**(projection 전 블록 feature vs
  projection 후 임베딩) 결정.
- **text**: CLIP text encoder를 `get_llm`이 받게. `AutoModel`이 CLIP을 주면 `CLIPModel`
  전체라 **text 부분만** 뽑는 처리 + `output_hidden_states`/pool 호환 확인.
- **권장 시작점**: `open_clip`으로 vision/text 인코더를 따로 얻어 각각 FeatureStore의
  이미지/텍스트 feature로 태우는 어댑터를 추가 (PAL의 "두 인코더" 모델과 가장 잘 맞음).
- **검증**: CLIP feature로 smoke 추출 → PAL anchor 학습이 도는지 (loss 수렴) + 기존
  체크포인트와 무관(새 인코더라 캐시 새로 생성).

> 주의: 인코더가 바뀌어도 anchor 체크포인트는 `class_name="PALAlignmentLayer"` 기반
> 로드라 호환. 단 **입력 차원(CLIP-ViT-B/16 = 768)** 이 바뀌므로 anchor `input_dim`
> 재설정.

---

## 5. CL 오케스트레이터 (핵심 신규 컴포넌트)

개념 (A 경로, naive 기준):
```
인코더 = frozen CLIP (한 번 로드)
anchor_state = None  (또는 무작위 초기화)
for t, task in enumerate([flickr, coco, pets, ...]):       # 8 task 순차
    1) task feature 추출/캐시  (extract_features, require_cached)
    2) anchor를 anchor_state에서 warm-start 하여 fit(task)   # ← warm-start 옵션 신규
    3) anchor_state = 학습된 anchor 체크포인트 저장
    4) 평가: task 1..t (또는 전체) retrieval 재평가 + HAVG + zs 6개  → 행렬[t]에 기록
집계: 평균 R@1, forgetting, BWT, zero-shot PD
```
필요 작업:
- **warm-start**: 현재 `_train_layer_pair`가 anchor를 `AlignmentFactory.create`로 매번
  새로 만듦 → "이전 task 체크포인트를 `load_alignment_layer`로 불러와 시작" 옵션 추가.
  (이게 "naive sequential fine-tune". 메소드 제안은 여기에 replay/정규화/anchor확장 등을
  얹는 자리.)
- **시퀀스 config**: task 목록 + 순서 + 공유 학습 하이퍼파라미터.
- **메트릭**: forgetting matrix `R[t][k]` = stage t 후 task k 성능. 거기서
  - 평균 성능(마지막 행), **Forgetting** = max_t R[t][k] − R[T][k] 평균,
  - **BWT/전이** (ε_j: 음수면 backward transfer), **zero-shot PD** = R0_zs − R_final_zs.

---

## 6. 작업 항목 (우선순위 / 단계)

| 단계 | 내용 | 산출물 | 비고 |
|---|---|---|---|
| **1** | CLIP ViT-B/16 인코더를 FeatureStore에 통합 | CLIP feature 추출 동작 | §4, 가장 먼저 |
| **2** | 8 task + HAVG + zs 6개 데이터셋 로더 | `get_datasets`에 신규 등록 | 데이터 준비(일부 비표준 = 큰 일) |
| **3** | anchor warm-start 옵션 (`_train_layer_pair`/fit) | task 이어학습 가능 | naive CL 골격 |
| **4** | CL 오케스트레이터 (시퀀스 + 평가 행렬) | `cli/`에 신규 스크립트 | train_subset 패턴 참고 |
| **5** | 메트릭 집계 (avg/forgetting/BWT/PD) + 결과 표 | 평가 리포트 | |
| **6** | (메소드 개발) naive 위에 본인 CL 전략 얹기 | — | 환경 완성 후 |

→ 1·3·4가 "환경 골격", 2가 "데이터", 5가 "측정". 메소드(6)는 그 다음.

---

## 7. 미결정 / 논의 포인트

1. **CLIP 표현 선택** — anchor 입력으로 CLIP의 (a) 블록별 토큰 feature(PAL 방식, layer
   selection 가능) vs (b) 최종 projection 임베딩(CLIP 정렬 공간) 중 무엇? → PAL의
   layer-selection/CAP를 살리려면 (a), C-CLIP과 직접 비교하려면 (b) 고려.
2. **평가 범위** — stage t에서 "task 1..t만" 재평가(seen) vs "전체 8개 매번"(미래 task
   포함, forward transfer 관찰). 논문 Figure 5는 0~8 전체 추적.
3. **warm-start 단위** — anchor 하나를 계속 이어가나(naive), task마다 새 anchor를
   추가/확장하나(메소드 여지). 벤치마크 골격은 naive로, 확장은 메소드에서.
4. **데이터셋 확보** — Lexica/Simpsons/Kream/Sketch는 비표준. 출처/라이선스/포맷
   확인 필요 (image-caption pair 형태로 변환).
5. **학습 하이퍼파라미터** — C-CLIP의 40ep/lr1e-6은 인코더 fine-tune용. anchor-only는
   재튜닝 (PAL 기존 config 기반).

---

## 8. 한눈 요약

- **그대로 차용**: 8-task 시퀀스, 3-트랙 평가(멀티모달 CL / zero-shot retrieval / zero-shot
  classification + PD), CLIP ViT-B/16 백본.
- **A 경로로 치환**: 인코더 frozen → forgetting/PD가 **anchor head 기준**으로 측정됨
  (PD는 0이 아님 — anchor가 zero-shot 경로에 관여).
- **재사용**: FeatureStore(per-task 캐시), PAL anchor, retrieval/zero-shot eval, fit.
- **신규**: CLIP 인코더 어댑터, 8+1+6 데이터셋 로더, anchor warm-start, CL 오케스트레이터
  + forgetting 매트릭스/메트릭.
- **메소드는 그 다음** — 이 환경 위에서 replay/정규화/anchor 확장 등 PAL-CL 전략 제안.
