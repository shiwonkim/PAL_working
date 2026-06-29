# src/ 디렉토리 재구성 — 진행 현황 & 계획 (한글)

> 최종 갱신: 2026-06-29 (HEAD `4a4c575`). 대부분 완료, 일부 남음.
> 목적: `src/` 디렉토리 이름이 역할을 직관적으로 드러내도록 재배치.

---

## 1. 재구성 동기 (원래 문제 3가지)

1. **진입점이 src/ 루트에 라이브러리와 섞여 흩어짐**
2. **`core/src/` 중첩** — 원본 레포 vendoring 흔적 (utils·datasets 중복)
3. **데이터 관련이 두 디렉토리로 분산** (+ utils 잡탕)

---

## 2. 현재 구조 (2026-06-29 기준, 실제)

```
src/
  datasets/     Dataset 클래스(coco/flickr/image_text/base) + data_utils
  features/     feature_store + feature_spec           (추출/캐시 파이프라인)
  models/
    encoders/   text_models + vision_models            (frozen 인코더)
    alignment/  pal + *_layer + alignment_factory + __init__(factory 초기화)
  training/
    base_trainer + alignment_trainer + clip_eval_trainer + csa_trainer
    loss/       clip_loss + siglip_loss
    optim/      optimizer + lars
  evaluation/   retrieval + zero_shot_classifier/consts + zero_shot_segmentation/patch_voting
  utils/        checkpoint, loader, metrics, plotting, base_factory, load_modules,
                measure_alignment, utils, train_utils
  (진입점, src 직속)  extract_features.py, train.py, train_alignment.py, train_subset.py
  (루트)             rerun_eval.py
```

---

## 3. 완료된 재구성 (커밋 순)

| 커밋 | 내용 |
|---|---|
| `e13e2ff` | `models/text/models.py` → `encoders/`(text+vision 대칭); `get_lvm` 본문을 `vision_models.load_lvm`으로 분리 |
| `de4659d` | datasets → `src/data/`(나중에 datasets) + 일회성 `prepare_*`/`vissl_download` 삭제 |
| `658256e` | `core/src/` 평탄화 → `utils/`(loader, plotting, train_utils) + `utils/optim/`. **src/core 완전 제거** |
| `4753c9c` | trainers + loss + optim + measure_alignment → `training/` |
| `763f2f8` | encoders + alignment → `models/` 하위로 묶음 |
| `5ba37dd` | measure_alignment → `training/`에서 다시 `utils/`로 |
| `2a5c064` | feature_store + feature_spec → `src/features/`로 분리 (utils 밖) |
| `4a4c575` | `src/data/` → `src/datasets/` 개명 (루트 `data/` 심볼릭과 혼동 회피) |

곁다리 정리(별도 커밋): lint 세트 제거(`958f998`), deepspeed dead 의존 제거(`ce335d0`),
pyproject/.gitattributes 제거(`adaf503`), Platonic-benchmark dead 코드 제거(`75481ab`),
`extract.py`→`extract_features.py` 개명(`cff92c8`), `paths.py` 삭제(`6908947`).

---

## 4. 원래 계획 대비 바뀐 결정 (중요)

| 항목 | 원래 계획 | 실제 결정 | 이유 |
|---|---|---|---|
| 인코더/정렬레이어 | `encoders/`, `alignment/` 각 1급 | **`models/{encoders,alignment}`** 로 묶음 | "신경망 = models" 응집 (사용자 선호) |
| 진입점 | `src/cli/`로 모으기 | **src/ 직속 유지** | 사용자가 cli/ 하위 원치 않음 (나중 개명만) |
| 데이터 디렉토리 | `src/data/` | **`src/datasets/`** | 루트 `data/`(심볼릭=실제 데이터)와 단어 충돌 |
| measure_alignment | `training/` | **`utils/`** | compute_score가 metrics 위 얇은 래퍼라 utils 성격 |
| loss/optim 위치 | (미정) | **`training/loss`, `training/optim`** | 학습 관련 = training 한 곳 |
| core utils.py | `dist_utils.py` | **`train_utils.py`** | 내용이 학습 인프라(clip_gradients/EarlyStopping/분산) |

---

## 5. 불변 조건 (재구성이 절대 깨면 안 됨) — 전부 유지됨

- **체크포인트는 `class_name` 문자열로 로드** (`checkpoint.load_alignment_layer` →
  `AlignmentFactory.create("PALAlignmentLayer")`). 모듈 경로 비의존 → 디렉토리 이동 안전.
  (models/ 이동 시 firefly forward sum=89.591690 동일 확인.)
- **factory 등록**: `models/alignment/__init__.py`의 `initialize_package_factory(__file__)`가
  자기 디렉토리를 스캔(`__file__` 기준 상대경로). alignment 레이어는 한 디렉토리에 모여 있어야 함.
- **클래스 이름 불변** (`PALAlignmentLayer` 등).

---

## 6. 검증 방법 (매 단계 반복)

- token smoke: `smoke_state_dict.yaml` → loss **3.0001** / val 5.7733
- CLS unpinned: `smoke_cls_unpinned.yaml` → 레이어 쌍 **(11,6)** score **0.3545** / loss 4.0650
  (compute_score = layer selection 경로 커버)
- (디렉토리/체크포인트 위험 시) firefly/galaxy 로드 + forward 값 비교
- (인코더 로더 변경 시) 캐시 비우고 extract → vision feature MD5 비교
- smoke 검증 config(`smoke_cls*.yaml`, `smoke_eval.yaml`)는 gitignore된 로컬 스크래치.

---

## 7. 남은 항목

1. **진입점 개명** (사용자 합의: src 직속 유지, 이름만) — 예:
   `extract_features.py`→`extract_feats.py`, `rerun_eval.py`→`src/eval.py`,
   `train_alignment.py`/`train_subset.py` 정리. (호출 경로 `python -m src.X` + run_pipeline.sh +
   feature_store 에러메시지 + docs 갱신 필요)
2. **`utils/utils.py` dead 함수 정리** — `set_seeds`, `walk_and_collect`, `set_requires_grad`,
   `has_batchnorms`, `get_available_torch_device` (정의만, 미사용; `set_requires_grad`/
   `has_batchnorms`는 train_utils에도 동일 기능 존재).
3. **`scripts/` 정리** (다음 작업 예정).
4. (선택) factory 인프라(`base_factory`+`load_modules`)를 `utils/factory/` 서브로.

---

## 8. 한눈 매핑 (옛 → 현재)

```
src/models/text/models.py        → src/models/encoders/text_models.py
(FeatureStore.get_lvm 본문)       → src/models/encoders/vision_models.py:load_lvm
src/core/src/datasets/*          → src/datasets/*
src/dataset_preparation/data_utils → src/datasets/data_utils.py
src/dataset_preparation/prepare_* → (삭제)
src/core/src/utils/loader        → src/utils/loader.py
src/core/src/utils/plotting      → src/utils/plotting.py
src/core/src/utils/utils         → src/utils/train_utils.py
src/core/src/optimizers/{utils,lars} → src/training/optim/{optimizer,lars}.py
src/trainers/*                   → src/training/*
src/loss/*                       → src/training/loss/*
src/measure_alignment.py         → src/utils/measure_alignment.py
src/utils/feature_store|feature_spec → src/features/*
src/alignment/*                  → src/models/alignment/*
src/encoders/*                   → src/models/encoders/*
src/utils/paths.py               → (삭제)
```
