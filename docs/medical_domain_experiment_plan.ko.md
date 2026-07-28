# 전문 도메인(의료) significance 실험 — 설계 & 구현 계획 (병리 + CXR)

**목적.** NeurIPS 2026 리뷰어 **2Mov W4**("*image-text pair-efficient training이 어디서
필요한지 불명확*")에 대한 답. general CLIP이 실패하고, 도메인 특화 멀티모달 사전학습
(BiomedCLIP/CONCH류)조차 대량 쌍이 필요한 전문 도메인에서, **PAL이 frozen 도메인 unimodal
인코더를 소량 쌍으로 정렬해 도메인 VLM에 필적**함을 실증한다. 정당성 계보: ASIF(NeurIPS'23),
FreezeAlign(CVPR'25), SAIL(CVPR'25), STRUCTURE(NeurIPS'25).

**두 도메인 병행**: 병리(승인 대기 → 문서화 후 대기) + **CXR(비게이트 → 먼저 진행)**.

---

## 1. 실험 시나리오 (Stage 1 + Stage 3; 3a·3b 둘 다)

학습은 **한 번**, eval만 데이터셋을 바꿔 두 번 (held-out / in-domain).

```
[Stage 1] 일반 모델 zero-shot (전제: large pretrained가 전문도메인서 실패)
  - general CLIP (ViT-L)
  - COCO-PAL (DINOv2 ViT-L + RoBERTa; 이미 보유) — general 정렬이 transfer 안 됨
  → 도메인 eval셋에 그대로 zero-shot (둘 다 실패 예상)

[Stage 3] 도메인 인코더: PAL(도메인 unimodal) vs 도메인 VLM
  - PAL = 도메인 영상 SSL 인코더 + 도메인 텍스트 인코더, 소량 도메인 쌍으로 학습
  - 상대 = 도메인 멀티모달 VLM (대량 쌍 사전학습)
  - 3a (held-out, 공정): PAL·VLM 둘 다 미학습 데이터로 zero-shot → 공정 비교
  - 3b (in-domain, 실전): PAL 학습셋 eval-split → PAL in-domain, VLM은 zero-shot
```

**평가 task = zero-shot 분류** (전 단계 통일). 병리=단일라벨 top-1 acc / CXR=다중라벨 AUC.

**baseline 스코프(확정)**: Stage 3는 **PAL vs 도메인 VLM만** (도메인 인코더 linear/mlp/fa/sail
대안정렬은 이번 스코프에서 제외).

### Stage별 증명 목표
| 단계 | 증명 |
|---|---|
| 1 | large pretrained(일반)가 전문도메인서 실패 + COCO 정렬은 transfer 안 됨 |
| 3a | 도메인 인코더 + 소량 쌍 PAL이 대량-쌍 도메인 VLM에 **공정하게** 필적 (핵심) |
| 3b | 소량 in-domain PAL이 off-the-shelf 대형 VLM을 **실전에서** 이김 (배포) |

---

## 2. 도메인 선택 제약

한 도메인에 {도메인 영상 SSL 인코더 + 도메인 텍스트 인코더 + 도메인 VLM} 세 개가 동시
공개돼야 성립 → **CXR**와 **병리(histopathology)** 둘만 통과. (망막 RETFound=VLM/쌍 약함,
CT 3D=복잡, "일반 biomedical"=BiomedCLIP 본진이라 PAL 도메인-인코더 우위 소멸.)

---

## 3. 병리 vs CXR 비교 (요약)

| 축 | 병리 | CXR |
|---|---|---|
| 영상 인코더 | **UNI/Virchow/Phikon** (최강 SSL, UNI=ViT-L DINOv2) | RAD-DINO (ViT-B) |
| 텍스트 | PubMedBERT (병리전용 PathologyBERT는 리포트체·소규모) | CXR-BERT-general / PubMedBERT |
| VLM 상대 | **CONCH**(병리특화) / PLIP / QuiltNet | BiomedCLIP(general biomedical) / CheXzero |
| eval | NCT-CRC/PCam **완전공개·top1-acc**(파이프라인 재사용) | NIH-CXR14 공개, **다중라벨 AUC**(신규 구현) |
| 학습 쌍 | Quilt-1M(공개, 접근요청) | OpenI(공개, 라벨추출) / MIMIC(credential) |
| legibility | 좋음 | **최상**(일반 리뷰어 각인) |
| 접근 마찰 | UNI/CONCH 게이트, Quilt 요청 | **전부 비게이트** |

**핵심 논리**: PAL 주장이 가장 세게 먹히는 조건 = (1) frozen 인코더 최강(UNI), (2) eval
깨끗·공개, (3) 상대가 도메인-특화 VLM. → 병리가 연구적으로 최강. CXR는 legibility + 즉시착수.

---

## 4. 확정 설계

### 4-A. 병리 (헤드라인, 승인 대기)
| 요소 | 헤드라인 | robustness 부록 |
|---|---|---|
| PAL 영상 | **UNI** (`hf-hub:MahmoodLab/UNI`) | Phikon-v2 (비게이트) |
| PAL 텍스트 | **PubMedBERT** (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`) | PathologyBERT |
| VLM 상대 | **CONCH** | PLIP (MIT, 비게이트) |
| Stage1 일반 | general CLIP ViT-L + COCO-PAL(보유) | — |
| 학습 쌍 | **Quilt-1M** (Zenodo 36GB, non-PMC + 품질필터 + subsample) | — |
| 3a held-out | **NCT-CRC-HE-100K + PCam** (top1-acc) | — |
| 3b in-domain | Quilt test (YouTube+Twitter) | — |

**텍스트 인코더 근거**: 영상 쪽이 도메인성을 짊어짐. 텍스트는 (a) PubMedBERT가 Quilt 캡션
(교육/문헌체)과 분포 일치 + 규모 큼 + 필드 표준 타워(BiomedCLIP/KEEP), (b) PathologyBERT는
임상 리포트체·34.7만 소규모라 캡션과 어긋남 → PubMedBERT 주, PathologyBERT는 "양쪽 다 전용"
부록. **CONCH/PLIP의 텍스트 타워는 금지**(joint 학습 → unimodal 전제 붕괴 + 순환).

### 4-B. CXR (즉시 착수, 전부 비게이트)
| 요소 | 선택 |
|---|---|
| PAL 영상 | **RAD-DINO** (`microsoft/rad-dino`, HF `Dinov2Model`, MSRLA 비게이트) |
| PAL 텍스트 | **CXR-BERT-general** (`microsoft/BiomedVLP-CXR-BERT-general`) 또는 PubMedBERT |
| VLM 상대 | **BiomedCLIP** (open_clip) [+ 선택 CheXzero/BioViL = 더 CXR-특화] |
| Stage1 일반 | general CLIP + COCO-PAL(보유) |
| 학습 쌍 | **OpenI (IU X-ray)** ~7.5k 영상-리포트 (공개, 라벨=CheXpert-labeler 추출) |
| 3a held-out | **NIH ChestX-ray14** (공개, 14라벨) |
| 3b in-domain | OpenI test split |

---

## 5. 데이터 누수 분석 & split 정책 (병리)

**결론: NCT-CRC·PCam(3a)=누수 0, Quilt(PAL 학습)=94% 깨끗, PMC 5.8%만 CONCH와 겹칠 수 있음.**

**Quilt-1M(101.7만) 구성 vs 학습셋:**
| 출처 | 쌍 수 | 비중 | CONCH(PMC-OA+EDU) | UNI(BWH/MGH WSI, 캡션無) |
|---|---|---|---|---|
| YouTube | 802,144 | 79% | ❌ | ❌ |
| Twitter/OpenPath | 133,511 | 13% | ❌ | ❌ |
| LAION-5B | 22,682 | 2% | ❌ | ❌ |
| **PMC-OA** | 59,371 | 5.8% | ⚠️ 가능 | ❌ |

**평가셋**: NCT-CRC(독일 NCT/Mannheim)·PCam(Camelyon16)은 UNI/CONCH **논문 자신들의
downstream 벤치** → 학습 미포함 구조적 보장 → 3a 공정.

**split 정책**: ① **3b Quilt test는 YouTube(+Twitter)만**(PMC/LAION 제외) → CONCH 진짜
zero-shot. ② (선택) PAL 학습에서도 PMC 제외 → 학습셋 소스 분리. ③ 3a 조치 불필요.

*(CXR 누수: NIH-CXR14는 BiomedCLIP이 PMC로 학습 → NIH 이미지가 PMC 논문그림에 있었을
가능성은 낮으나, OpenI가 PMC에 포함됐을 여지는 3b에서 명시 필요. RAD-DINO는 5개 공개 CXR로
학습 → OpenI/NIH 포함 여부 확인 필요.)*

---

## 6. 품질 필터링 계획 (Quilt, CSV 수령 후 확정)

Quilt lookup CSV 품질 컬럼 활용:
- **`not_histology`==True → 제거** (수동 품질체크 실패)
- **`subset`(source)** → 3b non-PMC 필터 + 노이즈 관리 (YouTube ASR 노이즈 最多)
- **medical entity 수 / caption 길이** → 저관련·초단문 캡션 제거
- **`corrected_text`** 우선 (ASR 보정본)
- `single_wsi` 등 부가 플래그 검토
→ 통과분에서 **subsample** (희소 스토리; 예: n=5k/50k 스케일).

---

## 7. 접근성 & 데이터 위치 상태 (2026-07-27 기준)

| 자산 | 상태 |
|---|---|
| UNI | ✅ HF **granted** (다운로드엔 이 머신 `huggingface-cli login` 필요) |
| CONCH | ✅ HF **granted** (eval용) |
| Quilt-1M | ⏳ **접근 요청 완료 — 승인 대기** (Zenodo 8239942, 512px 36GB) |
| PubMedBERT | ✅ **다운로드 완료** (공개, HF 캐시) |
| RAD-DINO / CXR-BERT / BiomedCLIP | 🔓 비게이트 → **즉시 다운로드 가능** |
| NCT-CRC / PCam | 👤 **사용자가 다른 서버에서 `data/`에 직접 배치** |
| OpenI / NIH-CXR14 | 🔓 공개 → 다운로드 필요 |

**데이터 위치 규칙**: `data/` 아래 **데이터셋명 폴더만** (별도 `pathology/` 하위폴더 X).
예: `data/quilt1m/`, `data/NCT-CRC-HE-100K/`, `data/pcam/`, `data/openi/`, `data/nih_cxr14/`.
모델은 HF 캐시(`~/.cache/huggingface`)에 저장(별도 data 폴더 불필요).

---

## 8. 코드 통합 맵 (정찰 결과)

**핵심 발견: PCam은 이미 완전 구현됨.** name→builder registry 없음(config 문자열로 추론).

| 확장 | 수정 위치 | 비고 |
|---|---|---|
| **UNI 영상** | config `lvm_model_name: "hf-hub:MahmoodLab/UNI"`; `src/models/backbones/vision_models.py:40` `if "vit" in name` 분기가 `hf-hub:` 문자열 미매치 → `hasattr(model,"blocks")`로 넓히고 `blocks.*.add_1` 노드명 확인 | timm 로드, CLS 있음 |
| **RAD-DINO 영상(CXR)** | `vision_models.py`는 **timm 전용** → RAD-DINO는 HF `Dinov2Model`이라 **별도 HF-vision 분기 필요**(또는 timm 호환 로드 탐색) | UNI와 다른 경로 |
| **PubMedBERT/CXR-BERT 텍스트** | config `llm_model_name`; `text_models.py`의 `AutoModel`이 raw BERT 수용 → 거의 그대로. tokenizer `padding_side=left` vs `pool_txt` 확인, dim 768(≠1024) 정렬 config 확인 | |
| **Quilt-1M 학습데이터** | `src/datasets/coco_dataset.py` 미러(신규 클래스); `data_utils.py get_datasets`에 분기 + `.name` 세팅; `train.py:54`/`eval.py:123` skip-tuple; config `features.dataset` | CSV→`df[image_path,captions]` |
| **PCam zero-shot** | ✅ **이미 완비** (`data_utils.py:478`, 메타 `:2385`/`:4396`) → `--zs pcam` + 데이터만 | |
| **NCT-CRC zero-shot** | `get_datasets`(ImageFolder) + `DATASETS_TO_CLASSES`(9클래스 알파벳순) + `DATASETS_TO_TEMPLATES` + `--zs` | 라벨순=클래스명순 필수 |
| **NIH-CXR14 zero-shot(CXR)** | `get_datasets` 분기 + 메타 + **다중라벨 AUC eval 경로**(현 파이프라인 top1-acc → 보강 필요) | |
| **VLM eval(CONCH/PLIP/BiomedCLIP/CLIP)** | open_clip 기반 **standalone 스크립트**(파이프라인 밖) | |
| **FeatureStore** | 변경 불필요 (`.name`/`.df`/tokenizer 계약만) | |

---

## 9. 구현 페이즈

| P | 내용 | GPU |
|---|---|---|
| **P0** | 데이터·모델 다운로드 (병리: UNI/CONCH/Quilt 승인후, PubMedBERT✅ / CXR: RAD-DINO/CXR-BERT/BiomedCLIP/OpenI/NIH) | X (네트워크) |
| **P1** | 인코더 로더 통합 (UNI timm 분기 / RAD-DINO HF-vision 분기 / 텍스트 BERT) + 피처추출 스모크 | 소 |
| **P2** | 데이터셋 등록 (Quilt / OpenI 클래스, NCT-CRC/NIH 등록) + 피처 캐시 | 중 |
| **P3** | PAL 학습 (도메인 쌍, subsample+필터) | 중 |
| **P4** | VLM standalone eval (CONCH/BiomedCLIP/PLIP/CLIP → 3a·3b) | 소 |
| **P5** | 통합 평가·표 (Stage1 + Stage3: 3a 공정/3b 실전) | 소 |

---

## 10. 진행 순서 (현재)

- **병리**: 설계·누수·필터·코드맵 **문서화 완료** → **Quilt 승인 + 이 머신 HF 로그인** 대기.
- **CXR 먼저 진행**: 전부 비게이트라 즉시 착수. 순서 = RAD-DINO/CXR-BERT/BiomedCLIP + OpenI/
  NIH-CXR14 다운로드 → RAD-DINO HF-vision 로더 통합 → OpenI 데이터셋 + 라벨추출 → NIH
  다중라벨 AUC eval → PAL 학습 → BiomedCLIP eval → 표.

## 11. 열린 결정 (CXR 착수 전)
1. CXR 텍스트 인코더: **CXR-BERT-general**(도메인) vs **PubMedBERT**(병리와 공유, 단순)?
2. eval task: **다중라벨 AUC**(표준, 신규구현) vs **retrieval**(OpenI 영상-리포트, 파이프라인 재사용)?
3. OpenI 라벨추출(CheXpert-labeler) 파이프라인 필요 여부(3b/3a 분류 시).
