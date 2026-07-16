# PAL 상대표현의 생성형 VLM 재료 가치 검증 — 실험 설계 (PoC)

> 상태: 제안(draft). 아직 미착수. 본 문서는 "PAL의 anchor 상대표현이 LLaVA식
> 생성형 VLM의 시각 입력(재료)으로 기능하는가"를 검증하기 위한 최소 실험 설계다.
> 기존 refactor/실험 문서와 달리 **미래 계획**이며, 착수 전 코드/자원 상황에 맞게
> 조정한다.

## 1. 동기

- Zero-shot 분류·retrieval은 CLIP류로 포화된 벤치라, "새 정렬 방법"이 거기서 몇 %
  이기는 것의 임팩트는 제한적이다.
- 이미지-텍스트 사전학습의 실질 가치는 점점 **LLaVA식 생성형 VLM의 시각 재료**로
  옮겨가고 있다. 그런데 **relative representation(상대표현)이 그 재료로서 가치가
  있는지 검증한 연구는 아직 없다.**
- PAL의 핵심은 **projection-free anchor 상대좌표**다(각 표현을 K개 learnable
  anchor에 대한 유사도 프로파일로 표현). 이는 Moschella et al. (ICLR'23,
  *Relative representations enable zero-shot latent space communication*)의
  아이디어를 **생성 세팅으로 확장**하는 셈이며, 성공 시 PAL의 가치를 "zero-shot
  수치"가 아니라 **"portable visual interlingua(이식 가능한 시각 공통언어)"**
  라는 새로운 축으로 끌어올린다.

## 2. 핵심 긴장(반드시 정면 대응)

LLaVA가 잘 되는 이유는 **spatial patch token(예: 576토큰 × 고차원)을 그대로**
LLM에 넣어 위치·OCR·카운팅 등 디테일을 보존하기 때문이다. 반면 PAL의 CAP 출력은
토큰을 K anchor로 pooling한 **(B, K) 전역 프로파일**이라:

- **공간 구조 소실** + **정보의 K차원 압축** → 분류/retrieval엔 충분하지만 **상세
  생성(캡셔닝/VQA)엔 정보 병목** 가능성이 높다.
- 따라서 "PAL 전역 프로파일 → LLM"의 순진한 버전이 표준 LLaVA에 디테일 태스크에서
  지는 것은 **실패가 아니라 예상된 결과**다. 설계는 이 병목을 분리·정량화하고, PAL이
  실제로 이길 수 있는 claim(=modularity)을 겨냥해야 한다.

## 3. 핵심 가설

- **H1 (재료 가능성)**: 각 패치 토큰의 anchor 유사도 프로파일 `(num_patches, K)`
  (= 공간 보존형 상대표현)을 시각 토큰으로 쓰면, LLM이 이를 디코딩해 유의미한 캡션/
  VQA를 생성할 수 있다.
- **H2 (병목 위치)**: 생성 품질 저하의 주원인은 "상대표현"이라는 성질보다 **전역
  pooling(공간 소실)** 이다. → per-token 상대표현이 전역 프로파일보다 크게 낫다.
- **H3 (modularity, 핵심 차별점)**: PAL 상대표현은 **인코더 불가지적 공통언어**로
  기능한다. 한 vision 인코더의 PAL 표현으로 학습한 projector를, **다른 vision
  인코더로 교체해도 재학습 없이(또는 최소 adapt로) LLM이 동작**한다. raw feature
  기반 projector는 이 교체에서 무너진다.

H3가 이 프로젝트의 진짜 승부처다. H1/H2는 H3를 위한 발판.

## 4. 방법: 공간 보존형 PAL 상대표현

- PAL의 anchor `A ∈ R^{K×D}`(학습됨)에 대해, vision 토큰 `x_i ∈ R^D` 각각을
  **anchor 상대 프로파일** `r_i = cos(x_i, A) ∈ R^K` 로 변환.
  → 이미지당 `R ∈ R^{P×K}` (P=패치수). CAP처럼 pooling하지 **않는다**(공간 유지).
- 이 `R`을 경량 projector(MLP)로 LLM 임베딩 차원에 매핑 → LLaVA식으로 텍스트
  토큰 앞에 prepend → LLM이 생성.
- 학습은 LLaVA 레시피(2-stage: projector pretrain → instruction tuning) 축소판.
  **vision 인코더와 PAL anchor는 동결**, projector(및 선택적으로 LoRA)만 학습.

## 5. 비교군(arms) — 병목을 분리해서 본다

| arm | 시각 입력 | 목적 |
|---|---|---|
| **A. raw-patch (표준 LLaVA)** | DINOv2 patch token `(P, D)` | 상한선/레퍼런스 |
| **B. PAL-global (병목 baseline)** | PAL CAP 전역 프로파일 `(1, K)` 또는 소수 토큰 | 압축·공간소실의 손실 측정 |
| **C. PAL-pertoken (제안)** | anchor 상대 프로파일 `(P, K)` | H1/H2 검증 |
| **D. raw-relative (대조)** | anchor 없이, 고정 랜덤/PCA 좌표 상대표현 | "학습된 anchor"의 기여 격리(선택) |

- A vs C: 상대표현의 절대 품질 격차.
- B vs C: 공간 보존의 효과(H2).
- **modularity(H3)**: 각 arm에서 vision 인코더 교체 실험(§6) 수행.

## 6. Modularity 프로토콜 (H3, 핵심)

1. Vision 인코더 `V1`(예: DINOv2 ViT-L)로 arm별 projector 학습.
2. Vision 인코더를 `V2`(예: SigLIP 또는 다른 ViT)로 **교체**. PAL anchor는 각
   인코더에 맞춰 이미 정렬돼 있다고 가정(또는 소량 재정렬).
3. **projector 재학습 없이** `V2` 표현을 통과시켜 생성 품질 측정. 추가로 "projector
   1-epoch 소량 adapt" 조건도 측정.
4. 지표: `V1→V2` 교체 시 품질 유지율. **C/D(상대표현)가 A(raw)보다 교체에 강건**하면
   H3 지지 — PAL 상대표현이 공통언어로 기능한다는 핵심 증거.

## 7. 모델 / 데이터 / 벤치마크 (PoC 스케일)

- **LLM**: 소형 우선 — Qwen2.5-1.5B/3B-Instruct 또는 Llama-3.2-1B/3B (env의
  transformers 4.45.2 호환 확인 필요). 스케일은 신호 확인 후 확장.
- **Vision**: DINOv2 ViT-L(주), 교체군으로 SigLIP/다른 ViT 1종.
- **학습 데이터**: LLaVA-style 축소판 — stage1 캡션 정렬(예: LLaVA-CC3M-595K 또는
  기존 COCO 캡션 재활용), stage2 instruction(예: LLaVA-Instruct 소량 subset).
- **평가 벤치**: 경량부터 —
  - 캡셔닝: COCO Caption(CIDEr) 소규모.
  - VQA: VQAv2/GQA subset, 또는 POPE(hallucination), MME(경량 종합).
  - **modularity 전용**: 위 벤치를 `V1`/`V2` 두 조건으로 측정한 유지율.

## 8. 성공 기준

- **최소 성공(H1)**: arm C가 무의미 출력이 아니라 **A 대비 일정 비율 이상의 캡션/
  VQA 품질**을 낸다(예: CIDEr ≥ A의 60–70%). "상대표현으로도 생성이 된다"의 증명.
- **핵심 성공(H3)**: `V1→V2` 교체 시 **C/D의 품질 유지율이 A보다 유의하게 높다**.
  → "PAL 상대표현 = 인코더 불가지적 재료" 주장 성립.
- **부분 성공도 논문거리**: A를 못 이겨도, **"상대표현은 품질 X만큼 손실하지만
  modularity Y를 얻는다"는 trade-off 정량화** 자체가 기여.

## 9. 단계별 스코프

- **Phase 0 — 타당성**: PAL per-token 상대표현 추출 파이프라인 + 최소 projector +
  소형 LLM으로 **캡션 한 배치라도 생성되는지** 확인(무의미/붕괴 여부).
- **Phase 1 — H1/H2**: arm A/B/C를 소규모 데이터로 학습·평가. 병목 위치 확인.
- **Phase 2 — H3(핵심)**: modularity 교체 실험. 여기서 신호가 이 프로젝트의 가치.
- **Phase 3 — 확장**: 스케일 업(LLM/데이터/벤치), arm D, 정식 벤치 리포트.

각 phase는 **다음으로 넘어가기 전 명확한 go/no-go**를 둔다(특히 Phase 0/1에서
붕괴하면 설계 재검토).

## 10. 리스크 & 대응

- **정보 병목으로 생성 붕괴** → K를 키우거나(anchor 수), per-token 유지, 상대+절대
  하이브리드(상대 프로파일 ⊕ 소량 raw) 실험.
- **LLaVA 학습 자체가 별도 프로젝트급 비용** → 반드시 소형 LLM/소량 데이터 PoC부터.
  전면 LLaVA 재현을 목표로 두지 않는다.
- **anchor의 인코더별 재정렬 필요성**(H3 전제) → V2에 대해 anchor를 소량 재정렬하는
  비용을 modularity 측정에 포함해 정직하게 리포트.
- **transformers 4.45.2 호환**(Qwen3 등 미지원 이력) → LLM 선택 시 사전 검증.

## 11. 관련 연구

- Moschella et al., *Relative representations enable zero-shot latent space
  communication*, ICLR 2023 — 상대표현의 공통언어 성질(본 프로젝트의 이론적 뿌리).
- Liu et al., *Visual Instruction Tuning (LLaVA)* — 생성형 VLM 표준 레시피/평가.
- 프로젝트 내 `ref_papers/`(SAIL, SOTAlign, STRUCTURE, FreezeAlign) — 정렬/probing
  관점의 선행 연구, 인코더 다양성 논거.

## 12. 한 줄 요약

**"PAL을 LLaVA에 꽂아 품질로 이긴다"가 아니라, "anchor 상대표현이 인코더
불가지적(portable) 생성 재료로 기능하는가"** 를 공간 보존형(per-token)으로 검증한다.
성공하면 PAL의 가치가 zero-shot 수치가 아닌 **portable visual interlingua**라는 새
축으로 올라선다.
