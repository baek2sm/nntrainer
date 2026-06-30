# YOLOv11m conv 트랙 — 위임용 작업계획서 (work orders)

> **목적**: conv 최적화를 **작게 쪼갠 독립 단위**로 외부(에이전트/개발자)에 위임하고, 우리는
> **정량 인수기준으로 결과만 회수**한다. 각 work order는 §0 공통 컨텍스트만 읽으면 단독 실행
> 가능하도록 진입점·구현지침·인수기준·측정법을 자기완결로 명시한다.
> 비-conv 위생(slice/attention/upsample/pool)은 **타 트랙**이라 여기 없음.
> 배경/근거: `yolov11_device_gap_analysis.md`, `yolov11_parity_masterplan.md`(상위 계획).

---

## 0. 공통 컨텍스트 (모든 work order 공유)

### 0.1 대상
- **모델**: YOLOv11m, 입력 832×832×3, nc=1. 총 ~57.2 GMAC/iter.
- **현 구성**: nntrainer, NCHW, Q4_0 weight + FP16 activation. ThreadManager(4 thread). conv 백엔드 = ggml i8mm **SMMLA** indirect GEMM(이미 SMMLA 사용 중).
- **기기**: Samsung SM-S926U (Galaxy S24+), SD8Gen3 (SMMLA/SDOT 지원). 측정은 **반드시 동일 1대**에서 A/B.
- **목표(전체 트랙)**: conv 868.7 → ~157ms/iter (ORT INT8 conv 동급). 본 계획서가 그 분해.

### 0.2 baseline 고정 (모든 단위 공통 출발)
- 각 work order는 **자기 base 브랜치에서 먼저 baseline을 재측정**(검출값·ms·RSS)하고, 그 값 대비 A/B를 보고한다. "이전 단위 추정치"가 아니라 **본인이 잰 base**가 기준.
- 참조 검출값(현 FP16+Q4_0): 박스 2개 confidence ≈ **0.9258 / 0.8868** (고양이 2마리, `input_cat.bin`). 정확 자리수는 base에서 캡처.

### 0.3 빌드·실행·측정 (재현 명령)
> ⚠️ **디바이스 측정은 3-에이전트 공유 자원이다. 측정 직전 반드시 락을 잡아라** —
> `DEVICE_MEASURE_PROTOCOL.md` 참조. acquire 없이 `yolov11_infer` 실행 금지.
> 요약: `device_lock/dev_acquire.sh <id> <task> <serial>` → ACQUIRED면 측정, BUSY면
> 비-device 작업 먼저 하고 5분 뒤 재시도(스킵 금지) → 끝나면 `dev_release.sh`.
```bash
# 빌드 (증분 ~3s)
ninja -C build_android
adb -s <serial> push build_android/jni/arm64-v8a/libnntrainer.so /data/local/tmp/yolov11/
# 실행 + 3중 계측 (8 iter 평균)
adb -s <serial> shell 'cd /data/local/tmp/yolov11 && LD_LIBRARY_PATH=. \
  YOLO_TENSOR_TYPE=FP32-FP16 YOLO_CONV_Q40=1 \
  YOLO_WEIGHTS=yolov11m_fw_q40_arm.safetensors YOLO_BENCH_ITERS=8 \
  YOLO_LAYER_PROFILE=1 YOLO_CONV_GEOM=1 YOLO_KERNEL_PROFILE=1 \
  ./yolov11_infer . input_cat.bin'
```
- **계측 env** (평소 0 오버헤드, env로만 ON):
  - `YOLO_LAYER_PROFILE` → `neuralnet.cpp` LayerProf: 레이어 타입/이름별 ms+개수
  - `YOLO_CONV_GEOM` → `conv2d_layer.cpp` ConvGeomProf: conv 카테고리별 ms+개수+**GMAC/s**
  - `YOLO_KERNEL_PROFILE` → `ggml_interface_fp16.cpp` ConvKernelProf: 3×3 커널 내부 gather+requant vs GEMM
- x86 유닛테스트(인프라 단위용): `meson test -C build_x86 <name>` 또는 해당 벤치 바이너리.

### 0.4 코드 진입점 맵 (conv)
| 영역 | 파일:라인 | 역할 |
|---|---|---|
| conv forward 분기 | `nntrainer/layers/conv2d_layer.cpp:704~771` | groups==1: ①1×1 `in_sub.transpose→act.dot`(720~725) ②Q4_0 `in_sub.convQ4_0Indirect`(744) ③FP `im2col+dot`(758~759) → **출력 transpose `765`** |
| depthwise/grouped | `conv2d_layer.cpp:802` (`depthwise_conv2d_fp32/fp16`), 816~ grouped else | dw 직접 커널 / per-channel im2col+GEMV |
| forward scratch | `conv2d_layer.cpp:561~595` (`finalize`), `683~695` | im2col/qgemm 버퍼 사전할당 |
| indirect gather | `nntrainer/tensor/cpu_backend/conv_indirect.h` | NCHW 입력 → 타일 gather (FP32/FP16 템플릿) |
| indirect GEMM(FP16) | `ggml_interface_fp16.cpp:565` `__ggml_q4_0_4x8_q8_0_indirect_GEMM_fp16` | gather 타일 → Q8_0 양자화 → SMMLA GEMM |
| GEMM dispatch(ARM) | `arm_compute_backend.cpp:383/390/407` | `__ggml_q4_0_4x8_q8_0_GEMM<>` / indirect |
| SMMLA 커널(NEON) | `nntr_ggml_impl/nntr_ggml_impl_neon.cpp:277`(gemm_4x8) `:697`(fp16) | inline asm `.inst …smmla` ×80 |
| **Q8_0 인프라(기존)** | `origin/feat/q8_0-stable`: `nntrainer/tensor/q8_0_tensor.{cpp,h}`, `test/unittest/unittest_nntrainer_q8_0_tensor.cpp` | 4D Q8_0 텐서+per-block dynamic quant. **cherry-pick 출발점** |

---

## 1. 검증 게이트 정의 (인수기준 공통 어휘)

| 게이트 | 정의 (이걸 충족해야 "완료") |
|---|---|
| **G-BIT** (비트동일) | 검출 confidence가 base와 **소수 6자리까지 동일**(box 개수·좌표 동일). 값 수학적 불변 단위(커널/레이아웃 교체)에 적용. |
| **G-TOL** (허용오차) | 검출 box **개수 동일** + 각 box **IoU ≥ 0.95** vs base + **\|Δconf\| ≤ 0.02**. int8 dtype 변경 단위에 적용. |
| **G-UNIT** (유닛테스트) | x86 단독 테스트 통과. 양자화 round-trip 또는 vs-FP 레퍼런스 상대오차 명시 bound 내. e2e 영향 없는 인프라 단위. |
| **공통 (모든 단위 필수)** | ① 전 레이어 출력 **finite**(NaN/Inf 0) ② 동일기기 **A/B ms/iter(8 iter)** 보고 ③ **RSS** 기록 ④ 목표 계측치(예: requant ms, transpose ms, GMAC/s)가 의도대로 변했음을 3중 계측 로그로 **증빙**. |

> 회귀(검출 게이트 실패 또는 비-finite)는 **즉시 fail**, 머지 금지. 속도가 나아져도 게이트 실패면 반려.

---

## 2. 작업 단위 요약표 + 의존 그래프

| ID | 작업 (최소단위) | 의존 | 게이트 | 대상/영향 |
|---|---|---|---|---|
| **A. FP 커널 교체 (완전독립·즉시·G-BIT)** |
| A1 | conv0 3×3 FP 스템 커널 교체 | — | G-BIT | conv0 48.7ms |
| A2 | FP 1×1 (3개) 커널 교체 | — | G-BIT | 13.25ms |
| A3 | depthwise 3×3 FP 벡터화 | — | G-BIT | 45.1ms |
| **B. int8/Q8_0 인프라 (G-UNIT, e2e 변화 0)** |
| B1 | Q8_0 4D 텐서 + per-block dynamic quant/dequant | — | G-UNIT | 인프라 |
| B2 | Q8_0×Q4_0 indirect GEMM int32누산 커널 정비 | B1 | G-UNIT | 인프라 |
| B3 | int8 SiLU (LUT) | — | G-UNIT | 인프라 |
| **C. W4A8 재양자화 제거 (NCHW 유지·G-TOL)** |
| C1 | 3×3 indirect: Q8_0 act 직접소비+int32누산+1회 requant | B1,B2 | G-TOL | 3×3 |
| C2 | 1×1-s1 matmul: int8 경로화 | B1,B2 | G-TOL | 1×1 371ms |
| C3 | 인접 quant conv 간 int8 persist | C1,C2 | G-TOL | 재양자화 132ms 본체 |
| C4 | 혼합정밀: detect head/attention FP16 유지 | C3 | G-TOL | 정확도 안전판 |
| C5 | depthwise int8화 | B1,A3 | G-TOL | depthwise |
| **D. NHWC transpose 제거 (FP16에서 G-BIT 먼저)** |
| D1 | NHWC 텐서/레이아웃 배선 + weight NHWC repack | — | G-UNIT | 인프라 |
| D2 | 1×1 conv NHWC (FP16) | D1 | G-BIT | 1×1 transpose |
| D3 | 3×3 indirect conv NHWC (FP16) | D1 | G-BIT | 3×3 transpose 142ms |
| D4 | depthwise NHWC (FP16) | D1 | G-BIT | dw |
| D5 | backbone conv chain NHWC (층간 transpose 제거) | D2,D3,D4 | G-BIT | chain |
| **D6. 결합 = parity maker (G-TOL)** |
| D6 | NHWC+int8 결합 → SMMLA 풀가동 | C3,D5 | G-TOL | conv→~157 parity |
| **E. 융합/튜닝 (초과)** |
| E1 | conv+requant+SiLU int8 epilogue 융합 | C3,D6 | G-TOL | |
| E2 | SMMLA 타일/스레드 튜닝 | D6 | G-BIT | conv→~130 |

**의존 그래프 (2갈래 병렬 → D6 합류):**
```
A1, A2, A3        (독립, 즉시 착수 가능, 병렬)
B1 → B2 → C1, C2 → C3 → C4        [재양자화 제거 갈래]
B1 → C5 ; B3 (독립)
D1 → D2, D3, D4 → D5              [transpose 제거 갈래, FP16 G-BIT]
        (C3) + (D5) → D6 → E1, E2  [합류 = parity → 초과]
```
**즉시 위임 가능(의존 0):** A1, A2, A3, B1, B3, D1.

---

## 3. 작업 단위별 work order (자기완결 사양)

> 각 항목 형식: **목표 / 범위(IN·OUT) / 진입점 / 구현지침 / 인수기준 / 측정 / 의존 / 산출물**.

### A1 — conv0 3×3 FP 스템 커널 교체
- **목표**: conv0(미양자화 3×3 stem)의 naive `__fallback_sgemm` 탈출로 48.7ms↓ (현 6.1 GMAC/s).
- **범위 IN**: groups==1 FP 경로(`conv2d_layer.cpp:758~759`)가 conv0에 한해 튜닝 커널 사용. **OUT**: 양자화 변경 금지, 다른 conv 무영향.
- **진입점**: `conv2d_layer.cpp:704~771` FP 분기; FP16 GEMM은 `custom_hgemm`(hgemm/, MLAS급) 존재 → 이걸로 라우팅.
- **구현지침**: conv0를 `custom_hgemm` 경로로 태우거나(권장) Q4_0 경로 편입. 값 불변 유지.
- **인수기준**: **G-BIT** + conv0 ms 측정치 하락(`YOLO_CONV_GEOM`의 `g1 fp 3×3 im2col+sgemm` 항목).
- **측정**: `YOLO_CONV_GEOM`/`YOLO_LAYER_PROFILE`, 동일기기 A/B.
- **의존**: 없음. **산출물**: 브랜치 + A/B 리포트(검출 6자리·ms·RSS).

### A2 — FP 1×1 커널 교체
- **목표**: FP 1×1 3개(13.25ms, 0.3 GMAC/s, naive sgemm) 튜닝 커널화.
- **진입점**: 동 FP 분기. 1×1은 im2col identity이므로 곧장 GEMM.
- **인수기준**: **G-BIT** + `g1 fp 1×1` ms 하락.
- **의존**: 없음. **산출물**: 동 양식.

### A3 — depthwise 3×3 FP 벡터화
- **목표**: depthwise(45.1ms, 1.7 GMAC/s, 스칼라) NEON 벡터화.
- **진입점**: `conv2d_layer.cpp:802` `depthwise_conv2d_fp32/fp16` (cpu_backend op).
- **구현지침**: per-channel 3×3을 행 단위 NEON FMA로. 값 불변.
- **인수기준**: **G-BIT** + depthwise ms 하락.
- **의존**: 없음. **산출물**: 동 양식.

### B1 — Q8_0 4D 텐서 + per-block dynamic quant/dequant
- **목표**: activation을 Q8_0(블록당 fp16 scale + 32×int8)로 양자화/역양자화하는 4D 텐서 인프라. ggml 블록 호환.
- **범위 IN**: 텐서 타입 + quantize_row/dequantize_row + 유닛테스트. **OUT**: conv 연결(=C1)·NHWC 무관.
- **진입점**: `origin/feat/q8_0-stable`의 `q8_0_tensor.{cpp,h}` + `unittest_nntrainer_q8_0_tensor.cpp` **cherry-pick** → 현 base에 빌드.
- **구현지침**: per-block **dynamic** scale(캘리브 불필요). int32 누산 전제(FP16 range wall 회피).
- **인수기준**: **G-UNIT** — quantize→dequantize round-trip 상대오차 ≤ (블록 최대값 기준 1/127 수준) 명시 bound; x86 빌드+테스트 PASS. e2e 무변화.
- **의존**: 없음(병렬 가능). **산출물**: 브랜치 + 유닛테스트 로그.

### B2 — Q8_0×Q4_0 indirect GEMM int32누산 커널 정비
- **목표**: Q8_0 activation × Q4_0 weight → int32 누산 → FP 출력. (커널은 이미 SMMLA 존재; **입력을 FP16 재양자화 없이 Q8_0로 직접** 받게 정비.)
- **진입점**: `ggml_interface_fp16.cpp:565` indirect GEMM 내부의 `__ggml_quantize_mat_q8_0_4x8` 단계 → Q8_0 입력이면 **이 양자화 스킵**. SMMLA core(`nntr_ggml_impl_neon.cpp:277/697`)는 그대로.
- **인수기준**: **G-UNIT** — 동일 입력에 대해 (FP→Q8_0→GEMM) vs (Q8_0 직접→GEMM) 결과 비트동일/허용오차; 레퍼런스 GEMM 대비 상대오차 bound.
- **의존**: B1. **산출물**: 브랜치 + 유닛테스트.

### B3 — int8 SiLU (LUT)
- **목표**: int8 입력/출력 SiLU(swish). FP 왕복 없이.
- **진입점**: SiLU/activation op (cpu_backend). 256-entry LUT 또는 int8 sigmoid+mul.
- **인수기준**: **G-UNIT** — int8 SiLU vs FP SiLU 상대오차 ≤ 명시 bound(예: 1 ulp@int8). 독립 테스트.
- **의존**: 없음. **산출물**: 브랜치 + 유닛테스트.

### C1 — 3×3 indirect: Q8_0 act 직접소비 + int32누산 + 1회 requant (NCHW)
- **목표**: 3×3 indirect conv가 Q8_0 activation을 직접 받아 int32 누산 후 **출력 1회만** requant. 매-conv FP→Q8_0 재양자화 제거.
- **범위 IN**: 3×3 indirect 분기만(`:744` 경로). NCHW·transpose 유지. **OUT**: 1×1·NHWC 변경 금지.
- **진입점**: `conv2d_layer.cpp:744` `convQ4_0Indirect` + B2 커널.
- **구현지침**: 입력 Q8_0(전 층 출력 또는 경계 1회 양자화) → gather(block 보존) → SMMLA int32 → bias+SiLU → 출력 Q8_0 requant. `OutputActivationDtype` 류 전파 필요.
- **인수기준**: **G-TOL** + `YOLO_KERNEL_PROFILE`의 gather+requant 132ms 항목 하락 증빙 + 전레이어 finite.
- **의존**: B1, B2. **산출물**: 브랜치 + A/B + 검출 IoU/Δconf 표.

### C2 — 1×1-s1 matmul: int8 경로화
- **목표**: 1×1 matmul 경로(371ms, 현재 act.dot·int8 아님)를 int8 GEMM으로.
- **진입점**: `conv2d_layer.cpp:720~725`(1×1 `transpose→act.dot`).
- **인수기준**: **G-TOL** + 1×1 conv ms/ GMAC/s 개선 + finite.
- **의존**: B1, B2. **산출물**: 동 양식.

### C3 — 인접 quant conv 간 int8 persist
- **목표**: 연속 quant conv 사이에서 출력을 FP로 되돌리지 않고 **int8 유지**(중간 dequant/requant 제거). lever #1의 본체.
- **범위 IN**: backbone의 conv→conv 체인. **OUT**: 민감층(C4가 처리).
- **진입점**: graph dtype 전파 + cast inserter(층간 Q8_0 유지). C1/C2 출력을 다음 conv가 직접 소비.
- **인수기준**: **G-TOL** + 재양자화 총량(132ms) 대폭 감소 증빙 + 전레이어 finite + RSS.
- **의존**: C1, C2. **산출물**: 동 양식.

### C4 — 혼합정밀 (민감층 FP16)
- **목표**: detect head/attention 등 민감층은 FP16 유지, backbone conv만 int8 → 정확도 안전판.
- **진입점**: 층별 dtype property; 경계 cast.
- **인수기준**: **G-TOL**(특히 검출 conf 회복) + 어떤 층을 FP16로 뒀는지 명시.
- **의존**: C3. **산출물**: 동 양식 + 층별 dtype 맵.

### C5 — depthwise int8화
- **목표**: depthwise를 int8 입력/누산으로.
- **진입점**: `conv2d_layer.cpp:802` dw 커널 + B1.
- **인수기준**: **G-TOL** + dw ms 개선 + finite.
- **의존**: B1, A3. **산출물**: 동 양식.

### D1 — NHWC 텐서/레이아웃 배선 + weight NHWC repack
- **목표**: conv가 channels-last activation을 소비하고, weight를 load-time에 NHWC(int8 repack 호환)로 재배치하는 인프라.
- **범위 IN**: 레이아웃 플러밍 + repack + 유닛테스트. **OUT**: 실제 conv 전환(=D2~)·int8 무관.
- **진입점**: 텐서 dim/stride, weight 로더, conv finalize(`:561~595`).
- **인수기준**: **G-UNIT** — NHWC 경유 GEMM 결과가 NCHW 레퍼런스와 등가(relayout 후 비교).
- **의존**: 없음(병렬). **산출물**: 브랜치 + 유닛테스트.

### D2 — 1×1 conv NHWC (FP16)
- **목표**: 1×1을 NHWC FP16로 — im2col/transpose 제거(채널 연속이라 GEMM 직행).
- **진입점**: `:720~725` + D1.
- **인수기준**: **G-BIT**(relayout 기준 비트동일) + 해당 conv transpose ms 소거 증빙.
- **의존**: D1. **산출물**: 동 양식.

### D3 — 3×3 indirect conv NHWC (FP16)
- **목표**: 3×3 indirect를 NHWC FP16로 — **출력 transpose(`:765`, 142ms) 제거**, gather가 채널연속이라 저렴.
- **진입점**: `conv_indirect.h` gather를 NHWC 어드레싱으로 + `:744`/`:765`.
- **인수기준**: **G-BIT** + transpose 142ms 항목(레이어 오버헤드) 소거 증빙.
- **의존**: D1. **산출물**: 동 양식.

### D4 — depthwise NHWC (FP16)
- **목표**: depthwise NHWC FP16.
- **진입점**: `:802` + D1.
- **인수기준**: **G-BIT** + 측정.
- **의존**: D1. **산출물**: 동 양식.

### D5 — backbone conv chain NHWC (층간 transpose 제거)
- **목표**: backbone conv 체인을 NHWC로 연결 → conv↔conv 사이 transpose 완전 제거. (비-conv 경계는 transpose/cast로 격리 — 타 트랙 인터페이스, 본 단위는 conv 내부만.)
- **진입점**: graph 레이아웃 전파; conv 섬 경계 transpose 노드.
- **인수기준**: **G-BIT** + 층간 transpose 총량 소거 + e2e ms.
- **의존**: D2, D3, D4. **산출물**: 동 양식.

### D6 — NHWC + int8 결합 (parity maker)
- **목표**: C(int8) 위에 D(NHWC)를 얹어 NHWC int8 GEMM = **SMMLA를 transpose/재양자화 오버헤드 없이 풀가동**. conv → ~157ms(ORT 동급).
- **진입점**: C3 + D5 통합.
- **인수기준**: **G-TOL** + conv GMAC/s가 65.8 → ~300+ 도달 증빙 + e2e ms + finite + RSS.
- **의존**: C3, D5. **산출물**: 동 양식 + ORT 직접 A/B(가능 시).

### E1 — conv+requant+SiLU int8 epilogue 융합
- **목표**: conv 출력 int32→requant→SiLU를 단일 int8 epilogue로 융합(메모리 왕복 제거).
- **인수기준**: **G-TOL** + epilogue ms 개선.
- **의존**: C3, D6. **산출물**: 동 양식.

### E2 — SMMLA 타일/스레드 튜닝 (SD8Gen3)
- **목표**: cache blocking·K-panel packing·스레드 밸런스로 conv 157→~130.
- **인수기준**: **G-BIT**(같은 수학) + ms 개선.
- **의존**: D6. **산출물**: 동 양식.

---

## 4. 위임 운영 규칙

1. **baseline 우선**: 각 단위는 base 브랜치에서 검출·ms·RSS를 먼저 캡처(§0.2). 회수 리포트는 항상 **A(base)/B(작업)** 쌍.
2. **리포트 양식(고정)**: ① 검출값(box별 conf, 게이트 판정 G-BIT/G-TOL) ② ms/iter(8 iter, 동일기기·serial 명시) ③ RSS ④ 목표 계측치 변화(requant/transpose/GMAC/s 로그 인용) ⑤ 전레이어 finite 여부.
3. **게이트 실패 = 반려**: 검출 회귀·비finite는 속도 무관 머지 금지.
4. **격리**: 한 단위는 자기 대상 conv-type/경로만 건드린다. 타 단위 경로 변경 금지(충돌 방지, 병렬 위임 가능).
5. **측정 신뢰**: §0.3 3중 계측 로그를 증빙으로 첨부. 2664ms류 타기기 수치와 직접비교 금지(동일기기만).
6. **커밋 규칙**: Signed-off-by/author 이메일은 `sb92.hong@samsung.com`만 사용(ax.samsung.com 금지).
7. **검증 baseline 브랜치**: 의미있는 진척마다 `yolov11-device-opt-baselinevN` 백업 push.
8. **장시간 빌드/디바이스 벤치**: 위임자가 직접(서브에이전트 동기위임 금지). 디바이스 동시 추론 세션 금지(OOM).
