# `src/` 파일 사용 현황 (extract → train → eval 파이프라인 기준)

> **경로 2026-06-29 갱신** — 디렉토리 재구성 반영 (HEAD `e8d4c38`).
> USED/UNUSED 판정은 2026-06-26 동적 캡처(`9ebf0bb`) 기준이며, 이후 바뀐 것은
> 경로/이름과 삭제분뿐. 큰 리팩터 후 다시 생성할 것.

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
`src/models/alignment/__init__.py`가 `initialize_package_factory`를 호출해
`src/models/alignment/` 아래 모든 모듈을 `importlib`로 로드(`@register`
데코레이터 실행) — 과 (2) `__init__.py` 부수효과를 놓침. 동적 캡처는 둘 다 잡음.

진입점 주의: `src/extract_features.py`, `src/train.py`, `rerun_eval.py`는 진입점이라
남을 로드하지만 자신은 실행 중 import되지 않으므로, 정의상 USED로 분류.

---

## ✅ 사용되는 파일

### 진입점 (CLI)
| 파일 | 작업 | 역할 |
|---|---|---|
| `src/extract_features.py` | **extract** | `run(extract_only=True)` — 인코더 → 캐시, 학습 X |
| `src/train.py` | **train** | `run(require_cached=True)` — 캐시만 읽고 학습 |
| `src/train_alignment.py` | extract+train | 공용 setup(`run` / `load_dataset`) + combined 실행 |
| `rerun_eval.py` | **eval** | 체크포인트 로드 → 독립 retrieval + zero-shot |

(repo 루트의 `run_pipeline.sh`가 이 세 단계를 체인으로 실행.)

### 핵심 파이프라인 (세 단계 전부)
| 파일 | 용도 |
|---|---|
| `src/training/alignment_trainer.py` | 허브: `prepare_features`(데이터) + `_train_layer_pair`(학습) + eval 메서드 |
| `src/training/base_trainer.py` | Trainer 베이스 (device, wandb init, lr finder) |
| `src/features/feature_store.py` | 캐시 경로 / 로드(mmap) / 추출 / dedup — extract·train·eval |
| `src/features/feature_spec.py` | `token_level` 정책 중앙화 (suffix / pool / layer) |
| `src/utils/checkpoint.py` | state_dict 직렬화/로드 (학습이 저장, eval이 로드) |

### Alignment 레이어 (팩토리가 `src/models/alignment/` 전체를 동적 등록)
| 파일 | 비고 |
|---|---|
| `alignment_factory.py`, `base_alignment_layer.py` | 팩토리 + 베이스 |
| `pal.py` | **PAL 레이어 — 실제 학습되는 것** |
| `linear_alignment_layer.py`, `mlp_alignment_layer.py`, `freeze_align.py`, `sail_star_mlp.py`, `cca_class.py` | 등록되지만 **대체** 레이어 — config `alignment_layer_name`이 PAL이 아닐 때만 사용 |

### Loss / 평가 / 인코더 / alignment-measure
| 파일 | 작업 | 역할 |
|---|---|---|
| `src/training/loss/clip_loss.py`, `siglip_loss.py` | train | CLIP / SigLip loss |
| `src/evaluation/retrieval.py` | eval | retrieval 지표 |
| `src/evaluation/zero_shot_classifier.py`, `consts.py` | eval | zero-shot 분류기 / 템플릿 |
| `src/utils/measure_alignment.py` | train | 레이어 선택 점수 (`compute_score`, mutual_knn) — 이 함수 하나로 trim |
| `src/models/encoders/text_models.py` | extract | LLM 로더 (`load_llm` / `load_tokenizer`) |
| `src/models/encoders/vision_models.py` | extract | 비전 인코더 로더 (`load_lvm`) |

### 대체 트레이너 (config 분기; extract/train 경로에서만 import)
| 파일 | 비고 |
|---|---|
| `src/training/clip_eval_trainer.py`, `csa_trainer.py` | config `clip:true` / `cca:true`일 때만; PAL 기본은 `AlignmentTrainer` |

### Datasets / utils / optim
| 파일 |
|---|
| `src/datasets/`: `data_utils.py` (get_datasets / transforms), `coco_dataset.py`, `flickr30k_dataset.py`, `image_text_dataset.py`, `base_dataset.py` |
| `src/utils/`: `utils.py`, `metrics.py`, `base_factory.py`, `load_modules.py`, `loader.py`, `plotting.py`, `train_utils.py` |
| `src/training/optim/`: `optimizer.py`, `lars.py` |

---

## ❌ 이 파이프라인이 안 쓰는 것 (단, 다른 용도로는 살아있음)

| 분류 | 파일 | 실제 성격 |
|---|---|---|
| **별도 eval 진입점** (segmentation — 다른 작업) | `src/evaluation/zero_shot_segmentation.py`, `zero_shot_patch_voting.py` | retrieval/zero-shot과 별개인 세그멘테이션 평가 |
| **별도 학습 진입점** | `src/train_subset.py` | 독립 subset 학습 스크립트 |
| **해석용 스크립트** | `viz/*.py` | repo 루트의 독립 figure/분석 스크립트. `src`를 라이브러리로 import; 파이프라인이 로드하지 않음 |

### 정리 과정에서 제거됨 (트리에 더 이상 없음)
- **Platonic 벤치마크 legacy** (2026-06-26): 옛 `extract_features.py`,
  `extract_token_features.py`, `src/utils/alignment_utils.py`, `src/models/tasks.py`
  제거; `measure_alignment.py`를 `compute_score`로 trim.
- **Dead util**: `src/utils/paths.py` (아무도 import 안 함).
- **데이터 준비 스크립트**: `dataset_preparation/prepare_*.py`, `vissl_download.py`
  (일회성 다운스트림 데이터셋 셋업).
- **scripts/**: 실험 런처(`vit*/`), `batch2_eval/`, 체크포인트 마이그레이션 도구
  (`migrate_checkpoints.py`, `verify_migration_roundtrip.py`, `verify_cache_suffix.py`,
  `verify_alignment_checkpoint.py`). `run_pipeline.sh`는 repo 루트로 이동.

---

## 관찰

1. 실제 파이프라인 코어는 **좁다**: 리팩터된 `alignment_trainer` /
   `feature_store` / `feature_spec` / `checkpoint`가 척추.
2. PAL이 아닌 5개 alignment 레이어는 **"등록되지만 미사용"** — 팩토리가 디렉토리
   전체를 import하지만 forward에 쓰이는 건 `pal.py`뿐.
3. "미사용" 파일들은 대부분 **다른 용도로 살아있음** (세그멘테이션 평가, subset
   학습, 해석용 viz) — 삭제 대상이 아니라 "이 파이프라인의 일부가 아닐" 뿐.
