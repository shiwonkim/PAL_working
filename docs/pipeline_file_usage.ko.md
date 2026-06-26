# `src/` 파일 사용 현황 (extract → train → eval 파이프라인 기준)

> **2026-06-26 기준** (HEAD `9ebf0bb`). 큰 리팩터 후 다시 생성할 것.

**feature 추출 → alignment 학습 → 평가** 파이프라인이 실제로 로드하는 `src/`
파일과 그렇지 않은 파일 정리.

## 판별 방법

grep 추측이 아님. 파이프라인 진입점 4개를 **실제로 실행**하고, 그 과정에서
import된 모듈을 `sys.modules`에서 캡처(동적 ground truth):

- `extract` / `train` / combined → COCO ViT-S smoke config로
  `python -m src.train_alignment` (`src/train_alignment.py`의 `run()`).
- `eval` → firefly 토큰 체크포인트로 `rerun_eval.py` (느린 eval 루프 없이
  import만 잡으려고 zs/rt를 비움).

순수 정적 import 그래프 BFS는 **불충분**함: (1) 팩토리 패턴 —
`src/alignment/__init__.py`가 `initialize_package_factory`를 호출해
`src/alignment/` 아래 모든 모듈을 `importlib`로 로드(`@register` 데코레이터
실행) — 과 (2) `__init__.py` 부수효과를 놓침. 동적 캡처는 둘 다 잡음.

진입점 주의: `src/extract.py`, `src/train.py`, `rerun_eval.py`는 진입점이라
남을 로드하지만 자신은 실행 중 import되지 않으므로, 정의상 USED로 분류.

---

## ✅ 사용되는 파일 (`__init__.py` 12개 포함 54개; 표에서는 단순 `__init__.py` 생략)

### 진입점 (CLI)
| 파일 | 작업 | 역할 |
|---|---|---|
| `src/extract.py` | **extract** | `run(extract_only=True)` — 인코더 → 캐시, 학습 X |
| `src/train.py` | **train** | `run(require_cached=True)` — 캐시만 읽고 학습 |
| `src/train_alignment.py` | extract+train | 공용 setup(`run` / `load_dataset`) + combined 실행 |
| `rerun_eval.py` | **eval** | 체크포인트 로드 → retrieval + zero-shot 독립 평가 |

### 핵심 파이프라인 (3작업 공통)
| 파일 | 어떻게 쓰이나 |
|---|---|
| `src/trainers/alignment_trainer.py` | 중심축: `prepare_features`(데이터) + `_train_layer_pair`(학습) + eval 메서드 |
| `src/trainers/base_trainer.py` | Trainer 베이스 (device, wandb init, lr finder) |
| `src/utils/feature_store.py` | 캐시 path / load(mmap) / 추출 / dedup — extract·train·eval |
| `src/utils/feature_spec.py` | `token_level` 정책 중앙화 (suffix / pool / layer) |
| `src/utils/checkpoint.py` | state_dict 직렬화/로드 (train 저장, eval 로드) |

### Alignment 레이어 (팩토리가 `src/alignment/`를 전부 동적 등록)
| 파일 | 비고 |
|---|---|
| `alignment_factory.py`, `base_alignment_layer.py` | 팩토리 + 베이스 |
| `pal.py` | **PAL 레이어 — 실제 학습에 쓰이는 것** |
| `linear_alignment_layer.py`, `mlp_alignment_layer.py`, `freeze_align.py`, `sail_star_mlp.py`, `cca_class.py` | 등록은 되지만 **대안 레이어** — config `alignment_layer_name`이 PAL이 아닐 때만 실제 사용 |

### 손실 / 평가 / 모델 / 정렬측정
| 파일 | 작업 | 역할 |
|---|---|---|
| `src/loss/clip_loss.py`, `siglip_loss.py` | train | CLIP / SigLip 손실 |
| `src/evaluation/retrieval.py` | eval | retrieval 메트릭 |
| `src/evaluation/zero_shot_classifier.py`, `consts.py` | eval | zero-shot 분류기 / 템플릿 |
| `src/measure_alignment.py` | train | layer selection 점수 (`compute_score`, mutual_knn) — 이 함수만 남기고 정리됨 |
| `src/models/text/models.py` | extract | LLM 로더 (`load_llm` / `load_tokenizer`) |

### 대안 trainer (config 분기, extract/train 경로에서만 import)
| 파일 | 비고 |
|---|---|
| `src/trainers/clip_eval_trainer.py`, `csa_trainer.py` | config `clip:true` / `cca:true`일 때만. PAL 기본은 `AlignmentTrainer` |

### 데이터 / 유틸 / core
| 파일 |
|---|
| `src/dataset_preparation/data_utils.py` (get_datasets / transforms) |
| `src/utils/`: `utils.py`, `metrics.py`, `base_factory.py`, `load_modules.py` |
| `src/core/src/datasets/`: `coco_dataset.py`, `flickr30k_dataset.py`, `image_text_dataset.py`, `base_dataset.py` |
| `src/core/src/optimizers/`: `lars.py`, `utils.py` · `src/core/src/utils/`: `loader.py`, `plotting.py`, `utils.py` |

---

## ❌ 이 파이프라인에서 사용 안 되는 파일

| 분류 | 파일 | 실제 성격 |
|---|---|---|
| **데이터셋 준비 스크립트** (일회성, 파이프라인 외) | `dataset_preparation/prepare_{aircraft,birdsnap,clevr,k700,kitti,memes,pets,resisc45,ucf101}.py`, `vissl_download.py` | 각 다운스트림 데이터셋 일회성 전처리. 평가 데이터 만들 때 손으로 실행 |
| **별도 eval 진입점** (segmentation — 다른 작업) | `evaluation/zero_shot_segmentation.py`, `zero_shot_patch_voting.py` | `scripts/batch2_eval/run_segmentation.sh`가 쓰는 세그멘테이션 평가. retrieval/zero-shot과 별개 |
| **별도 학습 진입점** | `train_subset.py` | 서브셋 학습용 독립 스크립트 |
| **죽은 유틸** | `src/utils/paths.py` | 어디서도 import 안 됨 (정적·동적 모두) |

### 2026-06-26 삭제됨 (Platonic 벤치마크 레거시, PAL은 안 씀)
`extract_features.py`, `extract_token_features.py`, `src/utils/alignment_utils.py`,
`src/models/tasks.py` 삭제 + `src/measure_alignment.py`를 `compute_score`만 남기고
정리. 이들은 다중 모델 "Platonic Representation" 추출 + 정렬 벤치마크 경로(`get_models`
model zoo, ViT+conv)로, PAL의 단일 인코더쌍 워크플로가 한 번도 안 씀.
`extract_token_features.py`는 `extract.py`(= `prepare_features`)로 대체된 얇은 래퍼.
실사용되는 `compute_score`(layer selection)는 유지.

---

## 핵심 관찰

1. 실제 파이프라인 코어는 **의외로 좁음**: 72개 중 실사용 ~42개, 그나마 절반은
   core/유틸. 리팩터로 정리한 `alignment_trainer` / `feature_store` /
   `feature_spec` / `checkpoint`가 중심축.
2. PAL이 아닌 alignment 레이어 5종은 **"등록되지만 미사용"** — 팩토리가
   디렉토리 전체를 import하나, forward에는 `pal.py`만 쓰임.
3. 확실한 **죽은 코드** 후보는 `src/utils/paths.py` 하나 (정적·동적 모두 안 닿음).
4. 나머지 "미사용" 파일은 대부분 **다른 용도로 살아있음** (데이터 준비,
   segmentation 평가, 레거시 추출, 서브셋 학습) — 삭제 대상이 아니라 "이
   파이프라인엔 안 쓰임".
