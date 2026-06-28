# src/ 디렉토리 전면 재구성 계획 (한글)

> 작성: 2026-06-26 (HEAD `6908947`). **계획 문서** — 아직 실행 안 함.
> 목적: `src/` 디렉토리 이름이 역할을 직관적으로 드러내도록 재배치.

---

## 1. 왜 재구성하나 — 현재 구조의 3가지 문제

1. **진입점이 src/ 루트에 라이브러리와 섞여 흩어짐**
   `extract_features.py` / `train.py` / `train_alignment.py` / `train_subset.py` /
   `measure_alignment.py`가 `alignment/` · `trainers/` 같은 디렉토리와 같은 레벨.
   "실행하는 것"과 "import되는 것"이 안 구분됨.

2. **`core/src/` 중첩 — vendoring 흔적**
   `src/core/src/datasets/...`, `src/core/src/utils/...` 처럼 "src 안의 core 안의
   src". 원본 레포를 통째 복사한 자국. `utils`가 두 곳, 데이터셋도 두 곳으로 갈림.

3. **데이터 관련이 두 디렉토리로 분산**
   `dataset_preparation/`(준비 스크립트 + `data_utils.py`) vs
   `core/src/datasets/`(실제 Dataset 클래스). "데이터 코드 어디?"가 두 곳.

---

## 2. 불변 조건 (재구성이 절대 깨면 안 되는 것)

체크포인트 로드 메커니즘을 확인한 결과(아래), 디렉토리 이동은 **안전**하되 2가지를
지켜야 한다:

- **체크포인트는 클래스 이름(`class_name`) 문자열로 로드된다.** `checkpoint.py`의
  `load_alignment_layer` → `AlignmentFactory.create("PALAlignmentLayer", ...)`.
  모듈 경로를 저장하지 않으므로 **파일을 옮겨도 기존 .pth 로드는 안 깨진다.**
  (firefly·galaxy 체크포인트 둘 다 new-format, `class_name="PALAlignmentLayer"` 확인함.)
  → **불변조건 ①: 클래스 이름을 바꾸지 말 것** (`PALAlignmentLayer` 등 유지).
- factory 등록은 `alignment/__init__.py`의 `initialize_package_factory(__file__)`가
  **자기 디렉토리를 스캔**해서 `@register`로 한다.
  → **불변조건 ②: alignment 레이어들은 한 디렉토리에 모아두고, 그 디렉토리
  `__init__.py`가 `initialize_package_factory`를 호출하게 유지** (흩뜨리면 등록이 깨짐).
  디렉토리째 옮기는 건 OK, 안에서 흩어놓는 건 금지.

---

## 3. 목표 디렉토리 구조

```
src/
  cli/          진입점(실행 스크립트)만
  data/         Dataset 클래스 + get_datasets + (prepare 스크립트는 서브로 격리)
  encoders/     인코더 로더 (LLM 등)
  alignment/    PAL 등 정렬 레이어 (그대로 — 불변조건 ②)
  features/     feature_store + feature_spec (추출/캐시 — 파이프라인 핵심)
  training/     trainers + loss + measure_alignment + optimizers
  evaluation/   retrieval + zero_shot (+ segmentation은 서브로)
  utils/        진짜 공용 유틸 + config 로더 + 체크포인트 + factory 인프라
```

원칙: **디렉토리 이름이 곧 역할**. "유틸"에 섞여있던 파이프라인 핵심(`feature_store`)을
`features/`로 올리고, 중첩된 `core/src/`를 평탄화한다.

---

## 4. 파일별 이동 매핑

### cli/  (진입점)
| 현재 | → 목표 |
|---|---|
| `src/extract_features.py` | `src/cli/extract_features.py` |
| `src/train.py` | `src/cli/train.py` |
| `src/train_alignment.py` | `src/cli/train_alignment.py` |
| `src/train_subset.py` | `src/cli/train_subset.py` |
| (루트) `rerun_eval.py` | 그대로 두거나 `src/cli/eval.py`로 (논의) |

> 주의: 진입점은 `python -m src.cli.train_alignment`처럼 호출이 바뀜 →
> `scripts/run_pipeline.sh`, `feature_store.py` 에러 메시지, docs 전부 갱신.

### data/  (데이터)
| 현재 | → 목표 |
|---|---|
| `src/core/src/datasets/base_dataset.py` | `src/data/base_dataset.py` |
| `src/core/src/datasets/image_text_dataset.py` | `src/data/image_text_dataset.py` |
| `src/core/src/datasets/downstream_tasks/coco_dataset.py` | `src/data/coco_dataset.py` |
| `src/core/src/datasets/downstream_tasks/flickr30k_dataset.py` | `src/data/flickr30k_dataset.py` |
| `src/dataset_preparation/data_utils.py` | `src/data/data_utils.py` |
| `src/dataset_preparation/prepare_*.py`, `vissl_download.py` | `src/data/prepare/` (일회성 스크립트 격리) |

### encoders/
| 현재 | → 목표 |
|---|---|
| `src/models/text/models.py` | `src/encoders/text_models.py` (평탄화) |

### features/
| 현재 | → 목표 |
|---|---|
| `src/utils/feature_store.py` | `src/features/feature_store.py` |
| `src/utils/feature_spec.py` | `src/features/feature_spec.py` |

### training/
| 현재 | → 목표 |
|---|---|
| `src/trainers/alignment_trainer.py` | `src/training/alignment_trainer.py` |
| `src/trainers/base_trainer.py` | `src/training/base_trainer.py` |
| `src/trainers/clip_eval_trainer.py` | `src/training/clip_eval_trainer.py` |
| `src/trainers/csa_trainer.py` | `src/training/csa_trainer.py` |
| `src/loss/clip_loss.py`, `siglip_loss.py` | `src/training/loss/` (서브로) — 또는 loss/ 독립 유지 (논의) |
| `src/measure_alignment.py` | `src/training/measure_alignment.py` (compute_score) |
| `src/core/src/optimizers/lars.py`, `utils.py` | `src/training/optimizers/` |

### evaluation/  (대부분 그대로)
| 현재 | → 목표 |
|---|---|
| `src/evaluation/retrieval.py`, `zero_shot_classifier.py`, `consts.py` | `src/evaluation/` (유지) |
| `src/evaluation/zero_shot_segmentation.py`, `zero_shot_patch_voting.py` | `src/evaluation/segmentation/` (서브로 격리) |

### alignment/  (그대로 — 불변조건 ②)
`alignment_factory.py`, `base_alignment_layer.py`, `pal.py`,
`linear_alignment_layer.py`, `mlp_alignment_layer.py`, `freeze_align.py`,
`sail_star_mlp.py`, `cca_class.py`, `__init__.py` → **이동 없음**.

### utils/  (공용만)
| 현재 | → 목표 |
|---|---|
| `src/utils/checkpoint.py` | `src/utils/checkpoint.py` (유지) |
| `src/utils/base_factory.py`, `load_modules.py` | `src/utils/` (유지 — factory 인프라) |
| `src/utils/metrics.py` | `src/utils/metrics.py` (유지) |
| `src/utils/utils.py` | `src/utils/utils.py` (유지, 공용 헬퍼) |
| `src/core/src/utils/loader.py` | `src/utils/loader.py` (config yaml 로더) |
| `src/core/src/utils/plotting.py` | `src/utils/plotting.py` |
| `src/core/src/utils/utils.py` | `src/utils/dist_utils.py` (이름 바꿔 충돌 회피 — 아래 주의) |

### 삭제될 빈 껍데기
`src/core/`, `src/core/src/`, `src/models/`, `src/dataset_preparation/`,
`src/trainers/`, `src/loss/` 등은 내용이 빠지면 `__init__.py`만 남으므로 제거.

---

## 5. 주의 / 결정 포인트 (실행 전 정할 것)

1. **`utils.py` 이름 충돌** — `src/utils/utils.py`와 `src/core/src/utils/utils.py`는
   **다른 파일인데 함수명이 겹친다**(`set_requires_grad`, `has_batchnorms`가 양쪽에).
   → 단순 통합 불가. core쪽을 `src/utils/dist_utils.py`(분산학습/`clip_gradients`/
   `EarlyStopping`/`save_checkpoint` 등)로 **이름 바꿔 분리**하는 안을 제안.
   (참고: `src/utils/utils.py`의 `set_requires_grad`/`has_batchnorms`는 dead라 별도
   정리하면 충돌 자체가 사라짐.)
2. **`loss/`를 `training/` 안에 넣을지 독립 유지할지** — 직관상 둘 다 가능. 제안은
   `training/loss/` 서브.
3. **`rerun_eval.py`(루트)를 `src/cli/eval.py`로 옮길지** — 루트 유지도 가능.
4. **`measure_alignment.py`** — 이제 `compute_score`(layer selection) 한 함수뿐이라,
   `training/`에 두거나 아예 `training/layer_selection.py`로 이름을 바꾸는 것도 고려.
5. **`prepare_*` 스크립트** — 대부분 일회성/미사용. `data/prepare/`로 격리하되 추후
   별도 정리(삭제) 대상.

---

## 6. import 경로 갱신 전략

- 모든 `from src.X import` / `import src.X`를 새 경로로 일괄 치환 (grep + sed).
- 규모: `src.core.src` 참조만 11개 파일 ~15곳 + 나머지. 단계별로 그 단계의 경로만 갱신.
- **클래스 이름은 절대 안 바꿈**(불변조건 ①). 디렉토리/모듈 경로만 변경.
- `alignment/__init__.py`의 factory 초기화는 그대로(불변조건 ②).

---

## 7. 단계별 실행 계획 (위험 낮은 것부터, 각 단계 독립 커밋 + 검증)

각 단계 후 **반드시 검증**:
- smoke token (`smoke_state_dict.yaml`) → loss 3.0001
- smoke CLS unpinned (`smoke_cls_unpinned.yaml`) → (11,6) 0.3545 / 4.0650
- firefly·galaxy 체크포인트 로드 (rerun_eval 또는 load_alignment_layer 직접)

| 단계 | 내용 | 비고 |
|---|---|---|
| **1** | `core/src/` 평탄화 → `data/`, `utils/`(loader/plotting/dist_utils), `training/optimizers/` | vendoring 중첩 제거, 가장 명백 |
| **2** | 진입점 → `src/cli/` | 호출 경로 변경 (run_pipeline.sh, 에러메시지, docs) |
| **3** | `feature_store`/`feature_spec` → `src/features/` | utils에서 핵심 분리 |
| **4** | `trainers`+`loss`+`measure_alignment` → `src/training/` | |
| **5** | `models` → `encoders/`, `dataset_preparation` → `data/` 통합, evaluation 서브정리 | 마무리 |

---

## 8. 위험 / 롤백

- **위험**: import 경로 광범위 변경 → 한 곳 빠뜨리면 ImportError. 각 단계 `python -c
  "import ..."` + smoke로 즉시 검출.
- **체크포인트**: class_name 기반이라 안전하지만, 각 단계에서 firefly/galaxy 로드를
  실제로 한 번 돌려 확인 (verify-before-claiming).
- **롤백**: 각 단계가 독립 커밋이므로 문제 시 해당 커밋만 revert.
- **원본 STRUCTURE와의 diff**: 이 repo는 code-only refactor copy라 구조 변경 OK.
  단 논문 리비전에서 원본과 대조 시 diff가 커진다는 점 인지.

---

## 부록: 현재 → 목표 한눈에

```
현재                          목표
src/extract_features.py    → src/cli/extract_features.py
src/train*.py              → src/cli/train*.py
src/measure_alignment.py   → src/training/measure_alignment.py
src/core/src/datasets/*    → src/data/*
src/core/src/optimizers/*  → src/training/optimizers/*
src/core/src/utils/loader  → src/utils/loader.py
src/core/src/utils/plotting→ src/utils/plotting.py
src/core/src/utils/utils   → src/utils/dist_utils.py (rename)
src/dataset_preparation/*  → src/data/ (+ data/prepare/)
src/models/text/models.py  → src/encoders/text_models.py
src/utils/feature_store    → src/features/feature_store.py
src/utils/feature_spec     → src/features/feature_spec.py
src/trainers/*             → src/training/*
src/loss/*                 → src/training/loss/*
src/alignment/*            → (그대로)
src/evaluation/*           → src/evaluation/ (+ segmentation/ 서브)
src/utils/{checkpoint,base_factory,load_modules,metrics,utils} → (그대로)
```
