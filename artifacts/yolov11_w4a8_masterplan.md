# YOLOv11m W4A8 마스터플랜 — ONNX parity까지

> ⚠️ **상태: 디테일 프로파일 이전 버전(lever #1 W4A8 심층 분석).** 이후 디바이스 정밀
> 프로파일로 비-conv 287ms 트랙이 새로 드러나 계획이 갱신됨 → **최신 계획은
> `yolov11_parity_masterplan.md`(4단계), 진입점은 `README.md`.** 이 문서는 W4A8(lever #1)
> 배경/근거의 심층 자료로 보존한다.

작성: 2026-06-29 / base: opt-stack (= PR #4017, CI green, #3999·#4000 머지 대기)
근거: 3중 병렬 분석 수렴 (우리 conv 코드 비용모델 / ONNX Runtime INT8 아키텍처 / origin/feat/q8_0-stable 인프라 재고)

---

## 0. 목표와 격차

| | 값 |
|---|---|
| 현재 (opt-stack/v17, S23U) | **~1000ms**, RSS ~226MB, 검출 0.926037/0.887984 |
| 목표 (ORT INT8, S23U 실측) | **247ms** |
| 격차 | **~4×** |
| 본진 | conv2d ≈ 시간의 **72%** (프로파일에 따라 55~72%) |

질문: "우리가 결국 더 빠르려면 어디까지 최적화가 이뤄져야 하는가?"
답: **NHWC int8 레이아웃 + 층간 int8 persist + int32 누산 + 층당 1회 requant + SMMLA 풀가동** = ORT의 데이터플로우를 복제하는 지점까지.

---

## 1. 왜 우리가 느린가 — 3중 분석 수렴

독립적으로 돌린 3개 분석이 **동일한 레버 순위**로 수렴했다:

| 순위 | 레버 | 크기 | 우리 현재 상태 |
|---|---|---|---|
| **#1** | **persistent int8 / quantize-once** | **가장 큼 (~격차 절반)** | ❌ 매 conv마다 FP32 gather + 전체텐서 FP32→Q8_0 재양자화, 출력은 다시 FP32로 다음 층 전달 |
| **#2** | **NHWC (이미 쓰는 SMMLA를 오버헤드 없이 풀가동)** | 큼 | ⚠️정정: SMMLA(i8mm)는 GEMM 코어가 **이미 사용 중**(LLM prefill FC와 동일 커널, ~430 GMAC/s). 문제는 NCHW dataflow — 출력 transpose 142ms + 층마다 activation gather/repack·재양자화. NHWC가 이걸 제거해 SMMLA를 풀로 먹임 |
| **#3** | **indirect conv (im2col 실체화 제거)** | 중 | ✅ **이미 완료** (conv_indirect.h, RSS −23.5%, **속도 flat**) |
| **#4** | **fusion (Conv+BN fold, Conv+SiLU, Conv+Add, QDQ elision)** | 가장 작음 | ✅ 대부분 완료 (BN-fold, SiLU 융합) |

### 핵심 통찰 (재프레이밍)
- 우리가 지금까지 한 일(#3 indirect, #4 fusion)은 **ONNX 분석상 속도 레버가 아니다**. 증거: ORT는 indirect conv 없이(MLAS는 im2col+GEMM) 247ms에 도달한다. 우리 indirect 전환도 RSS만 잡고 속도는 flat이었다 — 정확히 일치.
- **진짜 4×는 #1(quantize-once)과 #2(NHWC+i8mm)에 있다.** 지금까지 메모리/RSS는 잡았지만 **속도 레버는 손도 안 댄 상태.**
- ORT가 우리보다 빠른 본질: 입력 이미지를 **1회** int8로 양자화 → 층간 int8 유지 → int32 누산 → 층당 1회 requant. **활성값에 대한 float 왕복이 그래프에 아예 없다.** 우리는 층마다 FP32 im2col(4× 대역폭) + 전체텐서 FP32→int8 변환을 낸다.

---

## 2. 정직한 도착점 — 얼마나 가야 하나

- **W4A8(#1)만으로는 247ms 못 간다.** 효과 = gather가 int8(1B) 되어 대역폭 4×↓ + 양자화 단계 제거. conv의 gather/quant가 메모리·스칼라 바운드이므로 conv 시간 **~30~50%↓** 추정 → **~1000ms → ~700ms대**. (주의: 우리 indirect 전환이 속도 flat이었다는 사실은 "im2col write"가 병목이 아니었음을 뜻함 → 병목은 gather-read + scalar quant. W4A8가 바로 그 둘을 친다.)
- **247ms parity = #1 + #2 둘 다.** NHWC로 reduction dim을 연속화해야 SMMLA가 풀로 돌고, 그래야 GEMM 자체가 ORT급이 된다.
- 도착점 정의: **"NHWC, 층간 Q8_0 persist, int32 누산, 층당 1회 requant, SMMLA 풀가동"**.

---

## 3. 출발점 — origin/feat/q8_0-stable 재고

**있음 ✅**
- `Q8_0_Tensor` (4D, ggml 호환 block: fp16 scale + 32×int8 = 34B/block, byte-offset 정확)
- Q8_0 GEMM 커널: scalar fallback + AVX2 + ARM NEON (Q4_0의 int8 dot core 공유)
- ComputeOps `quantize_row_q8_0`/`dequantize_row_q8_0` API
- 모델레벨 dtype enum `WQ40A80`("Q4_0-Q8_0"), `WQ80A32`, `WQ80A16`
- CausalLM 층 통합 (FC/Embedding/TieWordEmbedding/Addition), Qwen3 Q4_0-Q8_0 테스트 PASS, CI green

**없음 ❌ (YOLO에 필요한 바로 그것)**
- **Conv2D Q8_0 경로 전무** (conv2d_layer.cpp 무수정, indirect-conv Q8_0 변형 없음)
- 범용 `OutputActivationDtype` 층 property (CausalLM은 dtype를 graph 구성 시 하드코딩)
- conv out_dim Q8_0 전파 / Q8_0 cast inserter
- YOLO 앱 Q8_0 wiring

**⚠️ 함정**: q8_0-stable의 CausalLM 층 통합은 `dequant Q8_0→FP32 → FP32 GEMM → requant→Q8_0`이다. 이건 **저장 포맷이지 perf 모델이 아니다**(FP32 compute라 오히려 느림). YOLO conv W4A8가 속도 win이 되려면 **진짜 int8 compute**(Q8_0 입력 → int4×int8 GEMM → int32 누산 → requant, FP32 우회 금지)여야 한다. 다행히 conv는 이미 int4×int8 indirect GEMM 커널(`__ggml_q4_0_4x8_q8_0_indirect_GEMM`)을 보유 → FC보다 유리한 출발.

---

## 4. 단계별 로드맵

### Phase 0 — 정밀 device 재프로파일 (가정 검증)
- 현 conv 내부를 gather / quantize / GEMM / epilogue로 실측 분해 (메모리의 "quantize 67% / GEMM 23%"는 일부 stale, 코드상 gather+quant 합산일 가능성).
- baseline 고정 (속도·RSS·검출). 이후 모든 GATE의 기준.
- **결정 입력**: W4A8 단독 기대치가 ~30% 수준이면 Phase 1+2를 한 흐름으로 묶을지 판단.

### Phase 1 — W4A8 numerics, lever #1 (NCHW 유지)
층간 활성을 Q8_0로 persist. 한 conv:
입력 Q8_0(전층이 양자화) → int8 gather(block 보존) → int4×int8 GEMM → int32 누산→FP32 → bias+SiLU(FP32) → **epilogue에서 출력 Q8_0 requant**.
- Conv2D Q8_0 입력 소비 + Q8_0 출력 경로 (conv2d_layer.cpp:656~690, 798~824)
- `OutputActivationDtype` property (layer_devel.h) + graph dtype 전파 + cast inserter
- int8 gather 변형 (conv_indirect.h) — **block 구조 보존이 난점**
- **GATE**: 검출 bit-similar(허용오차 내), 속도/RSS 측정, 전레이어 finite

### Phase 2 — NHWC 전환, lever #2 (247ms의 핵심)
- conv 경로 channels-last 전환 → reduction dim 연속 → SMMLA 풀가동
- weight load-time repack (SDOT/SMMLA interleave; MNN/llama.cpp 방식)
- **GATE**: SMMLA 실제 활용 확인(perf counter/디스어셈), 속도 점프 확인

### Phase 3 — 최종 fusion·정리
- int8→float→int8 경계 잔존 제거, SiLU int8 LUT(선택), slice int8화
- **GATE**: 247ms parity 판정

---

## 5. 핵심 리스크 / 미결 결정

1. **Q8_0 block 구조 vs 공간 gather** — Q8_0는 32-elem block(K 방향)에 scale. conv의 K=C×KH×KW. NCHW에선 공간 gather가 block을 깬다 → **NHWC가 사실상 #1의 전제**. Phase 1을 NCHW로 먼저 할지, 곧장 Phase 2와 묶을지가 최대 설계 결정.
2. **양자화 방식** — ggml식 **per-block dynamic Q8_0** 채택 권장: calibration 불필요, ORT per-tensor static보다 정확, **int32 누산이 FP16 range wall(W4A16 실패 원인) 회피**. (ORT는 per-tensor static+캘리브 — 우리는 굳이 따를 필요 없음.)
3. **Base 전략** — q8_0-stable를 base로 rebase할지(인프라 재사용, 단 conv는 신규) vs opt-stack 위 신규. q8_0-stable의 conv 부재 + FP32-compute 함정 고려.
4. **#4017 의존성** — #3999/#4000 머지 전엔 #4017이 안 나감. W4A8는 그 위에 쌓이는 구조라 base 흐름 영향 받음.

---

## 6. 한 줄 요약
**indirect conv·fusion으로 메모리는 잡았으나 속도 레버는 미착수. 4×의 본체는 (1) 층간 int8 persist로 per-conv 재양자화·FP32 gather 제거 + (2) NHWC로 SMMLA 풀가동. W4A8는 (1)이며 단독으론 ~700ms, 247ms parity엔 (2)까지 필수.**
