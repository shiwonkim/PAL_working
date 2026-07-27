# 전문 도메인(의료) significance 실험 설계 — 병리 vs CXR

**목적.** NeurIPS 2026 리뷰어 **2Mov W4**("*It is unclear where image-text pair-efficient
training is necessary...*")에 대한 답. general CLIP이 실패하고, 도메인 특화 멀티모달
사전학습(BiomedCLIP류)조차 대량 쌍이 필요한 전문 도메인에서, **PAL이 frozen 도메인
unimodal 인코더를 소량 쌍으로 정렬해 도메인 VLM에 필적**함을 실증한다.

관련 배경: `docs/laion_reimplementation_TODO.md`가 아니라 rebuttal 맥락 — frozen
unimodal encoder post-hoc alignment 계보(ASIF NeurIPS'23, FreezeAlign CVPR'25,
SAIL CVPR'25, STRUCTURE NeurIPS'25)가 데이터/컴퓨트 효율의 정당성을 이미 확립.

---

## 1. 실험 시나리오 (확정: Stage 1 + Stage 3, 3a·3b 둘 다)

학습은 **한 번**, eval만 데이터셋을 바꿔 두 번 (held-out / in-domain).

```
[Stage 1] 일반 모델 zero-shot (전제: large pretrained가 전문도메인서 실패)
  - general CLIP (ViT-L)
  - COCO-PAL (DINOv2 ViT-L + RoBERTa; 이미 보유) — general-domain 정렬이 transfer 안 됨을 보임
    → 도메인 eval셋에 그대로 zero-shot (둘 다 실패 예상)

[Stage 3] 도메인 인코더: PAL(도메인 unimodal) vs 도메인 VLM
  - PAL = 도메인 영상 SSL 인코더 + 도메인 텍스트 인코더, 소량 도메인 쌍으로 학습
  - 상대 = 도메인 멀티모달 VLM (대량 쌍 사전학습)
  - 3a (held-out, 공정): PAL·VLM 둘 다 학습에 안 쓴 데이터로 zero-shot → 공정 비교
  - 3b (in-domain, 실전): PAL 학습셋의 eval-split → PAL in-domain, VLM은 zero-shot
       + (방탄) VLM도 같은 데이터로 linear-probe/adapt → in-domain vs in-domain 공정화
```

**평가 task = zero-shot 분류** (모든 단계 동일 task로 통일; 도메인에 따라 top-1 acc 또는 AUC).

### Stage별 증명 목표
| 단계 | 증명 |
|---|---|
| 1 | large pretrained(일반)가 전문도메인서 실패 + COCO 정렬은 transfer 안 됨 → in-domain 쌍 필요 |
| 3a | 도메인 인코더 + 소량 쌍 PAL이 대량-쌍 도메인 VLM에 **공정하게** 필적 (핵심 펀치라인) |
| 3b | 소량 in-domain PAL이 off-the-shelf 대형 VLM을 **실전에서** 이김 (배포 시나리오) |

---

## 2. 도메인 선택을 결정하는 제약

한 도메인에 **세 가지가 동시에 공개**돼야 실험 성립:
1. 강한 도메인 **unimodal 영상** 인코더 (PAL의 frozen-encoder 전제)
2. 도메인 **unimodal 텍스트** 인코더
3. 도메인 **multimodal VLM** (상대 baseline)

이 조건을 통과하는 도메인은 사실상 **CXR**와 **병리(histopathology)** 둘. (망막 RETFound는
VLM/쌍데이터 약함; CT 3D는 복잡 + 3D unimodal 인코더 미성숙; "일반 biomedical"은
BiomedCLIP 본진이라 PAL의 도메인-인코더 우위가 사라짐.)

---

## 3. 병리 vs CXR 비교

| 축 | **병리 (histopathology)** | **CXR (chest X-ray)** |
|---|---|---|
| 도메인 영상 인코더 | **UNI / Virchow / Phikon** — 의료 이미징 통틀어 최강 SSL (UNI = ViT-L, DINOv2, ~100k 슬라이드/~100M 패치) → PAL 전제 **극대화** | **RAD-DINO** (ViT-B, image-only, ~800k CXR) — 견고하나 규모 작음 |
| 도메인 텍스트 인코더 | PubMedBERT / BioClinicalBERT (병리-특화 BERT는 약하지만, 영상 쪽이 도메인성의 핵심) | **CXR-BERT-general** / PubMedBERT (반드시 *general* — specialized는 image-text 봄) |
| 도메인 VLM 상대 | **CONCH**(Nature Med'24), **PLIP**, **QuiltNet** — **병리 특화** | **BiomedCLIP**(사실 general biomedical, PMC-15M) / CXR 특화는 **CheXzero·BioViL·GLoRIA** |
| zero-shot eval | **NCT-CRC-HE-100K(9-class), PCam(binary), LC25000, SICAP** — **완전공개·credential 불필요**, **top-1 acc** → 현 파이프라인 eval 그대로 | CheXpert/MIMIC **credential 필요**, **AUC 프롬프트 eval 새로 구현** 필요 |
| 공개 학습 쌍 | **Quilt-1M**(100만 image-text, 공개), OpenPath | OpenI(~7.5k, 공개, 라벨추출 필요) / MIMIC(377k, credential) |
| "CLIP 실패" 선명도 | H&E 염색 조직 = 자연영상과 극도로 멀어 매우 선명 | 선명 |
| 리뷰어 legibility | 좋음(ML-for-med 커뮤니티서 CONCH/UNI/PLIP 확립) | **최상**(일반 ML 리뷰어에게 즉시 각인) |
| 텍스트 품질 | Quilt 캡션은 유튜브/트위터 유래 → 노이즈 있음 | 임상 리포트 → 깨끗 |
| 파이프라인 재사용 | **높음**(top1-acc zero-shot 그대로) | 낮음(AUC eval 신규) |
| 인코더 접근성 | UNI/CONCH는 HF **게이팅**(약관 동의, 빠름). Phikon/PLIP은 비게이트 | RAD-DINO/CXR-BERT/BiomedCLIP 대체로 공개; MIMIC/CheXpert는 credential |

### 핵심 논리 (왜 병리가 실험적으로 더 강할 수 있나)
실험 주장 = "**소량 쌍으로 frozen 도메인 인코더를 PAL 정렬 → 대형 도메인 VLM에 필적**".
이게 가장 세게 먹히는 조건:
1. frozen 인코더가 최대한 강함 → **UNI(ViT-L) > RAD-DINO(ViT-B)**
2. eval이 깨끗·공개 → **병리 패치분류(credential 0, top1-acc) > CXR(credential, AUC 신규)**
3. 상대가 **도메인-특화** VLM → **CONCH(병리특화) > BiomedCLIP(general biomedical)**

→ 세 축 모두 병리 유리. **"주관 개입"은 인코더 픽이 아니라 도메인 레벨(CXR vs 병리)에
있었음** — CXR 안에서라면 RAD-DINO+CXR-BERT+(CheXzero/BiomedCLIP)이 실제로 CXR 최선.
단, BiomedCLIP은 CXR 특화가 아닌 general biomedical이라 "특화 상대"로는 다소 느슨.

---

## 4. 도메인별 구체 조합

### (A) 병리 — 연구적으로 가장 강한 조합 (1순위 추천)
- 영상 unimodal: **UNI** (게이팅 걸리면 **Phikon**, 비게이트·ViT-B)
- 텍스트 unimodal: **PubMedBERT**
- 멀티모달 상대: **CONCH** (접근 난이 시 **PLIP**, 완전공개)
- 학습 쌍: **Quilt-1M** subsample (희소 regime 스토리)
- **3a** held-out zero-shot 분류: **NCT-CRC-HE-100K + PCam** (공개, 둘 다 미학습 → 공정)
- **3b** in-domain: Quilt test split retrieval (PAL in-domain, CONCH zero-shot)
- Stage 1 일반: general CLIP ViT-L + COCO-PAL(보유)

### (B) CXR — 안전·범용 legibility 조합
- 영상 unimodal: **RAD-DINO** (HF `Dinov2Model`)
- 텍스트 unimodal: **CXR-BERT-general** / PubMedBERT
- 멀티모달 상대: **BiomedCLIP** (또는 CXR 특화 CheXzero/BioViL — 더 타이트)
- 학습 쌍: **OpenI**(공개, ~7.5k, 라벨은 CheXpert-labeler로 추출) / MIMIC(접근 시)
- **3a** held-out zero-shot 분류: **NIH ChestX-ray14**(공개, 14라벨) 또는 CheXpert(등록)
- **3b** in-domain: OpenI test split (라벨 추출) 또는 학습셋 test
- Stage 1 일반: general CLIP ViT-L + COCO-PAL(보유)

---

## 5. 판단
- **"가장 강한 실험" 기준 → 병리 (A)**: PAL 전제 극대화(UNI) + eval 마찰 0(공개·top1-acc) +
  특화 VLM 상대(CONCH) + 파이프라인 재사용.
- **"안전 legibility" 기준 → CXR (B)**: 일반 리뷰어 각인 + 깨끗한 임상 텍스트.

---

## 6. 필요한 통합 작업 (공통 리프트)
1. **인코더 로더**: 도메인 영상 인코더가 timm이 아닌 HF/커스텀(UNI=timm-호환 ViT지만 가중치
   로드, RAD-DINO=HF `Dinov2Model`) → `load_lvm`(현재 timm ViT-only) 손질. 텍스트는 HF BERT →
   `load_llm` 대체로 수용.
2. **VLM eval**: CONCH/PLIP/BiomedCLIP/general CLIP은 open_clip/커스텀 → **standalone eval
   스크립트**(파이프라인 밖).
3. **데이터 다운로드**: 병리(Quilt-1M, NCT-CRC-HE-100K, PCam) 또는 CXR(OpenI, NIH-CXR14).
4. **zero-shot 분류 eval**: 병리=top1-acc(현 파이프라인 재사용) / CXR=프롬프트 AUC(신규).
5. (CXR 한정) **라벨 추출**: 리포트 → CheXpert-labeler로 14라벨.

## 7. 남은 결정 / 다음 스텝
1. **도메인 확정**: 병리(A) vs CXR(B).
2. **접근성 검증**: UNI/CONCH 게이팅, Quilt-1M 라이선스, 패치벤치 다운로드 (웹서치 확인 예정).
3. **3b 방탄**: VLM linear-probe adapt 포함 여부.
4. 확정 후 → 데이터/인코더 다운로드 + `load_lvm`/`load_llm` 손질 + eval 스크립트 착수.
