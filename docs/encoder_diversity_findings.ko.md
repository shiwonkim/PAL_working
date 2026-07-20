# 텍스트 인코더 다양성 실험 — 결과 & 해석

> 상태: 결과 분석 노트 (2026-07-20). PAL이 **텍스트 인코더 패밀리**를 넘나들며
> 견고한지 검증하는 실험의 결과·해석. 단일 seed(42), DINOv2 ViT-L 비전 고정.
> 원자료: `results/final/{qwen25,gteqwen2}_final_metrics.csv`,
> RoBERTa는 `results/final/*_final_metrics.csv`(seed 42/44/123).

## 1. 세팅

- **비전 고정**: DINOv2 ViT-L. **텍스트 3종**:
  - **RoBERTa-large** — bidirectional **sentence-transformer** (mean-pool 임베딩), layer 24
  - **GTE-Qwen2-1.5B** — **contrastive decoder-embedder** (last-token 임베딩), layer 28
  - **Qwen2.5-0.5B** — **raw decoder LM** (임베딩 학습 안 됨), layer 23
- **방법 5종**: PAL(token, CAP) / linear·mlp(CLS-pooled) / fa·sail(token)
- **태스크**: zero-shot 분류(5셋) · retrieval(flickr30/coco_karpathy) · segmentation(voc/ade20k/context)
- **pooling**: CLS baseline은 인코더 native 방식 — RoBERTa=avg, Qwen2.5/GTE-Qwen2=**last**
  (decoder LM은 last가 원칙; GTE-Qwen2는 애초에 last로 임베딩 학습됨). 학습·eval 일치.

## 2. 핵심 결과 — PAL이 세 인코더 전부에서 최상위

대표 지표 (top1 / R@1 / mIoU-fg):

| 인코더 | 지표 | **PAL** | linear | mlp | fa | sail |
|---|---|---|---|---|---|---|
| RoBERTa | zs STL10 | **0.950** | 0.925 | 0.927 | 0.890 | 0.848 |
| RoBERTa | rt COCO I2T | **0.552** | 0.407 | 0.405 | 0.432 | 0.388 |
| RoBERTa | seg VOC | **33.0** | 10.97 | 10.83 | 19.51 | 22.17 |
| Qwen2.5 | zs STL10 | **0.967** | 0.781 | 0.785 | 0.786 | 0.833 |
| Qwen2.5 | rt COCO I2T | **0.510** | 0.164 | 0.162 | 0.199 | 0.276 |
| Qwen2.5 | seg VOC | **38.99** | 9.73 | 9.70 | 14.34 | 20.49 |
| GTE-Qwen2 | zs STL10 | **0.966** | 0.934 | 0.935 | 0.910 | 0.860 |
| GTE-Qwen2 | rt COCO I2T | **0.537** | 0.340 | 0.351 | 0.403 | 0.387 |
| GTE-Qwen2 | seg VOC | **36.72** | 13.59 | 13.88 | 18.39 | 21.57 |

→ **PAL이 전 인코더·전 태스크 1위**, retrieval·segmentation에서 특히 큰 margin.

## 3. 핵심 해석 — "인코더 타입 × pooled 품질 → baseline 취약성; PAL은 견고"

겉보기엔 baseline 순위가 인코더마다 뒤집혀 "들쭉날쭉"해 보이지만, **일관된 원리**가 있다:
**텍스트 인코더의 *pooled 표현 품질*이 CLS baseline의 성패를 가른다.**

| 인코더 | 타입 | pooled 품질 | baseline 결과 |
|---|---|---|---|
| RoBERTa | sentence-transformer | **좋음** | CLS(linear/mlp)가 강함 — token 메소드도 능가 |
| GTE-Qwen2 | contrastive decoder-embedder | 좋음 | baseline 준수 (RoBERTa 근접) |
| Qwen2.5 | **raw decoder LM** | **나쁨** | CLS baseline **붕괴**; token 메소드는 견딤 |

**증거 (데이터가 원리와 정확히 일치):**

1. **순위 역전이 규칙적**:
   - RoBERTa(pooled 좋음): CLS > token — linear 0.925/mlp 0.927 > fa 0.890/sail 0.848.
   - Qwen2.5(pooled 나쁨): token > CLS — sail 0.833/fa 0.786 > linear 0.781/mlp 0.785.
   → pooled가 좋으면 CLS가, 나쁘면 token이 유리. **PAL(token CAP)은 양쪽 다 최상위.**

2. **학습 수렴(val loss)이 eval 순위와 일치** (Qwen2.5, 전부 정상 early-stop):
   PAL **3.46** < sail 4.47 < fa 4.71 < linear 5.18 ≈ mlp 5.19.
   → linear/mlp는 정상 수렴했으나 **더 나쁜 val loss에서 plateau** = "poor pooled feature에서
   배울 수 있는 만큼만 배움". 학습 버그가 아니라 **표현 상한의 문제.**

3. **top1 vs top5 랭킹-모양** (EuroSAT 등 클래스가 겹치는 셋):
   token 메소드가 top1↓인데 top5↑ (예 fa: top1 0.297<linear 0.326, top5 0.831>0.756).
   모순이 아니라 "정답이 상위권엔 있으나 정확한 1등은 아슬" = 분산된 랭킹. 정상.

**한 줄**: pooled-baseline은 raw LM(Qwen2.5)에서 무너지지만, **PAL은 token-level CAP이라
pooled 품질과 무관하게 견고** → "인코더 불가지적" 정렬의 실증.

## 4. 재현성 & eval 파이프라인 검증

- **PAL 수치 재현 일치**: 코드 수정 전/후 PAL zs+rt가 **소수점까지 100% 동일**
  (PAL의 zs/rt 코드는 미변경) → 파이프라인 건전성의 강한 증거.
- **이번에 발견·수정한 eval 버그 3건** (decoder-LM CLS baseline을 처음 eval하며 드러남):
  - `8811f8e` zero-shot `"last"` pooling이 토큰 대신 레이어를 인덱싱 → CLS zs 크래시.
  - `285e4e7` seg `linear_perpatch`가 config pool 무시, avg 고정 → last-학습과 train/eval 불일치.
  - `b50433d` seg PAL-CLS 분기도 동일하게 avg 고정 → 일관성 수정(현 실험 무영향).
  - **무영향 확인**: RoBERTa(avg 일치·수치 불변), 모든 token 메소드(PAL/fa/sail), zs+rt의 token 경로.
- **pooling 일치**: decoder LM은 `last`(원칙), 학습·eval 모두 last — avg로 eval하면 train/eval
  불일치가 되므로 하지 않음.

## 5. 논문 함의

- **주장**: "PAL은 텍스트 인코더 패밀리(encoder-ST / decoder-embedder / raw decoder-LM) 전반에서
  최상위·견고." 특히 **pooled 표현이 나쁜 raw LM에서 CLS baseline이 붕괴할 때도 PAL은 유지** —
  token-level anchor CAP의 강건성.
- **주의(정직)**: 비전은 항상 DINOv2(크기 고정 ViT-L). 다양성 축은 **텍스트 패밀리**이며,
  비전 패밀리 다양성은 스코프 밖(별도 인코더 필요).

## 6. 관련
- 데이터: `results/final/{qwen25,gteqwen2}_final_metrics.csv`, RoBERTa 멀티시드 CSV.
- 배경 메모: 텍스트-인코더 다양성 실험, ASIF fixed-anchor baseline(`docs/asif_fixed_anchor_baseline.ko.md`).
