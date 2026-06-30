# YOLOv11m 디바이스 격차 분석 — 우리 vs ONNX Runtime

> 진입점·배경·재현방법은 **`README.md`** 참조. 계획은 **`yolov11_parity_masterplan.md`**.
> 이 문서는 측정/분석 본체다. 원시 로그: `raw/our_profile_8iter.txt`·`raw/ort_perop_8iter.txt`.

측정일 2026-06-29 / 기기 **SM-S926U (Galaxy S24+, Snapdragon 8 Gen 3 / SM8650)** / 4 intra-op threads / 입력 `input_cat.bin` (832×832 실사진)

모든 수치 **동일 기기·동일 입력·동일 스레드 수**. ORT는 onnxruntime-android 1.23.2 (ORT_ENABLE_ALL).

---

## 0. 한눈에 — e2e 실측

| 버전 | 레이아웃 | 가중치/활성 | **지연** | RSS | 검출 |
|---|---|---|---|---|---|
| **우리** (opt-stack/v17) | NCHW | Q4_0 w + FP16 act | **~1440–1600ms** | 278MB | 0.9258 / 0.8868 ✓ |
| **ORT FP32** | NHWC | FP32 / FP32 | **841ms** | 547MB | (디코드 일치) |
| **ORT INT8** | NHWC | S8 w (per-ch) / S8 act (per-tensor) | **208ms** | 233MB | (디코드 일치) |

→ **우리 vs ORT INT8 = ~7× 느림.** 목표는 247ms가 아니라 **이 기기 실측 208ms**.

### ★ 가장 중요한 단일 사실 (2026-06-29 갱신 — 근본 원인 확정)
**apples-to-apples FP32 비교: 우리 FP32-FP32 = 3130ms / RSS 553MB vs ORT FP32 = 841ms / 547MB = 우리가 3.7× 느림.**
양자화를 아예 안 해도(양쪽 다 FP32, 같은 메모리) 3.7× 느리다 → 격차는 양자화와 **무관**.

**근본 원인 = FP32 GEMM이 naive 폴백 커널이다.**
- 우리 FP32 conv: `Conv2DLayer::forwarding` (groups==1) → im2col(ThreadManager 병렬, memcpy 최적화됨) → `filter_kernel.dot()` → `sgemm_fp32` → (build_android에 `USE_BLAS` 미정의) → **`__fallback_sgemm`**.
- `__fallback_sgemm`(fallback_internal.cpp:201) = **naive 삼중 루프**: `double` 누산, 스칼라 곱(NEON/FMA 없음), 단일 스레드, 캐시 블로킹/패킹 없음.
- 코드베이스 GEMM 현황: **FP32 = naive 폴백만(유일하게 튜닝 부재)** / FP16 = `custom_hgemm`(hgemm/ 8×16 블록+패킹, MLAS급) / Q4_0·Q8_0 = i8mm indirect GEMM(ThreadManager). 프로젝트가 곧장 FP16·양자화 커널로 가서 FP32 sgemm엔 투자 안 함.
- 그래서 Q4_0-FP16(1450ms)가 FP32-FP32(3130ms)보다 2.2× 빠름 — 전자는 튜닝 i8mm, 후자는 naive 폴백.
- ORT FP32: MLAS(패킹+블로킹+NEON FMA+멀티스레드 sgemm). 우리 naive 폴백 vs MLAS = 그 3.7×.

**결론: '양자화 부족' 서사는 틀렸다. 우선 FP32 GEMM부터 제대로 만들면(혹은 튜닝 경로로 라우팅) apples-to-apples로 ORT FP32를 따라잡을 수 있고, 양자화는 그 위에 쌓인다.**

---

## 1. Per-op 분해 (per-iteration, ORT 프로파일 JSON 집계)

### ORT INT8 — 208ms
| op | ms/iter | % | count/iter |
|---|---|---|---|
| **QLinearConv** | **156.7** | **75.7%** | 113 |
| QLinearMul (SiLU 곱) | 15.0 | 7.2% | 103 |
| QLinearSigmoid (SiLU) | 13.6 | 6.5% | 102 |
| QLinearConcat | 8.9 | 4.3% | 26 |
| Resize (upsample) | 5.7 | 2.7% | 2 |
| QLinearSoftmax / Add / MatMul / Transpose / Split … | 나머지 ~8ms | ~3.6% | |

### ORT FP32 — 841ms
| op | ms/iter | % |
|---|---|---|
| **Conv** | **765** | **90.6%** |
| QuickGelu (SiLU) | 55.6 | 6.6% |
| Concat | 9.9 | 1.2% |
| 나머지 | ~15ms | ~1.6% |

### 핵심 비교
- ORT **conv만**: FP32 765ms → INT8 157ms = **4.9× (양자화를 제대로 했을 때의 conv 가속)**.
- 우리 conv 추정: e2e ~1450ms × 72% ≈ **~1040ms**. ORT INT8 conv 157ms 대비 **~6.7× 느림**.
- SiLU: ORT INT8에서 Sigmoid+Mul 합 ~29ms(14%) — 무시 못 할 2위지만 conv가 압도적.

---

## 2. ORT INT8가 빠른 메커니즘 (cloned 소스 + 프로파일로 확정)

1. **NHWC 레이아웃** (`NhwcTransformer`가 그래프 전체를 channels-last로 변환) → conv reduction dim(채널)이 메모리 연속 → MLAS의 SMMLA/i8mm 커널 풀 throughput.
2. **층간 int8 persist** — 입력 1회 양자화 → `QLinearConv`가 int8 입력 소비, int32 누산, **층당 1회만** 출력 requant → 다음 층도 int8. **활성값 float 왕복이 그래프에 아예 없음.**
3. **QDQ fusion** — Quant/Dequant 노드가 `QLinearConv`/`QLinearMul`/`QLinearSigmoid`로 흡수됨 (프로파일에 DequantizeLinear 0.6ms로 사실상 0).
4. indirect conv(MlasConvSym)는 메모리 절약용이지 속도 본질 아님 — ORT는 indirect 없이도 빨랐을 것 (우리 indirect 전환이 속도 flat이었던 것과 일치).

## 3. 우리가 느린 메커니즘 (마스터플랜 #1+#2, 실측 확증)

- **매 conv마다**: FP32 im2col gather(4B/elem, 대역폭 4×) + 전체 텐서 FP32→Q8_0 재양자화 + 출력 다시 FP32 → 다음 conv가 또 재양자화. = **층마다 float 왕복 + 재양자화**.
- ★**정정(2026-06-29): GEMM 명령은 이미 SMMLA다.** 우리 conv indirect GEMM 코어(`nntr_gemm_q4_0_4x8_q8_0[_fp16]`, NEON inline asm `.inst …smmla` ×80)는 LLM prefill FC와 **동일한 SMMLA(i8mm) 커널**을 쓴다. 실측이 이를 확증: 커널 내부 GEMM만 116ms/~50GMAC ≈ **430 GMAC/s로 ORT 전체 conv 365보다 빠름**. 따라서 conv 5.5× 격차는 "SDOT→SMMLA 교체"가 **아니다**. (decode GEMV `nntr_gemv_*`만 SDOT인데, M=1이라 SMMLA 타일을 못 채우므로 그게 정상.)
- 진짜 격차는 GEMM **바깥의 dataflow**: ① NCHW 출력 transpose 142ms/iter + ② 매 conv FP→Q8_0 재양자화/gather 132ms/iter. NCHW라 activation을 층마다 다시 gather·repack해야 하고, 출력을 [OH·OW,oc]→[oc,OH·OW]로 되돌려야 함. **NHWC의 역할 = SMMLA를 켜는 게 아니라, 이미 빠른 SMMLA를 transpose/재양자화 오버헤드 없이 풀로 먹이는 것.**
- 결과: weight는 4bit, GEMM은 SMMLA인데 activation **데이터플로우**(NCHW float 왕복+재양자화)가 conv를 메모리·스칼라 바운드로 만든다.

---

## 4. 도착점과 레버 (확정)

목표 = **208ms** (이 기기 ORT INT8). conv를 ~157ms로, 나머지 ~50ms.

| 레버 | 기대 효과 | 상태 |
|---|---|---|
| **#1 층간 int8 persist + quantize-once** (W4A8) | conv float왕복/재양자화 제거. 단독으론 ~700–900ms대 추정 | 미착수 |
| **#2 NHWC** (SMMLA는 이미 사용 중) | 출력 transpose 142ms 소거 + activation channel-last로 층간 gather/repack 제거 → 이미 ORT급인 SMMLA GEMM을 오버헤드 없이 풀로 먹임. #1과 합쳐 247→208ms parity | 미착수 |
| #3 indirect conv | RSS −23.5%, 속도 flat | ✅ 완료 (속도 레버 아님) |
| #4 fusion (BN/SiLU) | 작음 | ✅ 대부분 완료 |

**정직한 결론**: 우리가 지금까지 한 일(#3, #4)은 메모리만 잡았고 속도 레버는 미착수. 7× 격차의 본체는 #1+#2. #1만으론 ORT FP32(841ms) 근처까지가 한계일 수 있고, 208ms parity엔 #2(NHWC+SMMLA)가 필수.

## 4b. ★ 디테일 프로파일 — 우리 전체 연산 분해 (2026-06-29, 디바이스 실측)

측정: SM-S926U, FP32-FP16+Q4_0 weight, `YOLO_BENCH_ITERS=8`, 3개 인스트루먼트(`YOLO_LAYER_PROFILE`/`YOLO_CONV_GEOM`/`YOLO_KERNEL_PROFILE`). 모든 수치 **per-iteration**(÷8). e2e=1210.9ms/iter, layerprof 합=1161.8ms(나머지 ~49ms=디코드/NMS).

### (A) 레이어 타입별 (우리)
| 타입 | ms/iter | % | 개수/iter | ORT INT8 대응 | ORT ms | **격차** |
|---|---|---|---|---|---|---|
| **conv2d** | **868.7** | **74.8%** | 112 | QLinearConv 113 | 156.7 | **5.5×** |
| **slice** | **119.7** | 10.3% | 22 | Split 11 | 0.7 | **★171×** |
| **psa_attention** | **98.4** | 8.5% | 1 | (Q)MatMul+Softmax 4 | ~3.2 | **★31×** |
| **upsample2d** | **53.7** | 4.6% | 2 | Resize 2 | 5.7 | **9.4×** |
| **pooling2d** | **14.8** | 1.3% | 3 | (Nhwc)MaxPool 3 | 0.2 | **★74×** |
| concat | 4.9 | 0.4% | 26 | QLinearConcat 26 | 8.9 | 0.55× (우리 승) |
| addition | 1.5 | 0.1% | 19 | QLinearAdd 21 | 1.6 | ~1× (동등) |
| SiLU | (conv에 융합됨) | — | — | QSigmoid+QMul 205 | 28.6 | — |
| **TOTAL(layer)** | **1161.8** | | | | 207 | **5.85× (e2e)** |

### (B) conv 세부 분해 — 커널크기·종류·경로별 (우리, convgeom)
| 카테고리 | 개수/iter | ms/iter | %conv | **GMAC/s** | 비고 |
|---|---|---|---|---|---|
| **g1 quant 3×3 indirect [a16]** | 44 | **390.1** | 44.9% | **87.7** | i8mm **SMMLA** indirect GEMM (커널만 떼면 ~430 GMAC/s=ORT급), 본체 |
| **g1 quant 1×1-s1 matmul [a16]** | 57 | **371.1** | 42.7% | **60.8** | act.dot(weight), 3×3보다 MAC당 느림 |
| g1 fp 3×3 im2col+sgemm [a16] | 1 | 48.7 | 5.6% | **6.1** | conv0 스템(미양자화)=naive `__fallback_sgemm` |
| depthwise 3×3 [a16] | 7 | 45.1 | 5.2% | **1.7** | detect-head dw, 산술강도 낮음 |
| g1 fp 1×1 im2col+sgemm [a16] | 3 | 13.25 | 1.5% | **0.3** | naive sgemm, 최악 효율 |
| **conv TOTAL** | 112 | 868.7 | 100% | **65.8** | vs ORT INT8 **365 GMAC/s** |

총 MAC = **57.2 GMAC/iter** (YOLOv11m@832 이론치와 일치 → MAC 계산 검증됨).

### (C) 3×3 indirect 커널 내부 (kernelprof, 44 conv/iter)
| 구간 | ms/iter | 비고 |
|---|---|---|
| gather + Q8_0 재양자화 | 132.0 | 매 conv FP→Q8_0 재양자화(W4A8면 제거) |
| i8mm GEMM | 116.2 | 실제 연산 |
| **커널 합** | **248.2** | |
| **레이어 오버헤드(390.1−248.2)** | **~141.9** | 출력 transpose([OH·OW,oc]→[oc,OH·OW]) + bias + SiLU = **NHWC면 transpose 소거** |

### ★ 결론 — 어디가 병목이고 어디가 이미 이상적인가
1. **5.85× e2e 격차는 두 부류로 분해된다:**
   - **(a) conv 5.5× (868→157)** — 단, **GEMM 명령은 이미 SMMLA(i8mm)고 커널만 떼면 ~430 GMAC/s로 ORT급**(§3 정정). 격차는 GEMM 바깥 dataflow: 재양자화 132ms + 출력 transpose 142ms. **레버=W4A8 quantize-once(#1, 재양자화 소거)+NHWC(#2, transpose 소거+SMMLA를 오버헤드 없이 풀로 먹임)**. 어려운 본진이지만 "SMMLA 커널 신규작성"이 아니라 "SMMLA를 막는 NCHW dataflow 제거"가 본질.
   - **(b) 비-conv 스칼라 디스패치 군 ~287ms/iter (24%)** — slice 119.7+attention 98.4+upsample 53.7+pool 14.8. ORT는 이걸 **합쳐 ~12ms**에 한다(10~171× 격차). **W4A8/NHWC와 무관, 독립적이고 더 쉬운 win**(getValue/setValue 스칼라 → memcpy/벡터화, numerics 불변). 이것만 잡아도 1211→~935ms.
2. **이미 이상적(건드릴 것 없음):** concat(우리가 ORT보다 빠름), addition(동등).
3. **놀라운 발견:** 1×1 conv 경로(371ms)가 3×3 indirect(390ms)와 맞먹는 2위이며 MAC당 더 느리다(60.8 vs 87.7 GMAC/s). conv0 fp 스템(48.7ms, naive sgemm)은 단발 quick-win.
4. **우선순위 재정렬:** 기존 마스터플랜은 (a)만 봤다. (b)의 287ms는 "위생(hygiene)" 성격의 더 쉬운 레버로 별도 트랙. 정직한 도착점: (b) 처리 → ~935ms, (a) W4A8+NHWC → 200ms대.

---

## 4c. ★ ONNX lowering 검증 — 저장 그래프 vs 실제 실행 op 개수 (2026-06-29)

**의문**: ORT는 그래프 최적화(QDQ fusion, NHWC 변환, op 제거)를 하므로 **저장된 모델의 op 개수 ≠ 실제 실행 개수**. 저장 모델을 세면 과대계상된다. → 세 가지를 모두 떴다.

검증: 디바이스 프로파일의 **node 시간 합 / model_run 시간 = 98.7%(INT8) / 99.7%(FP32)**. 나머지 ~1%는 ORT 스케줄링 오버헤드. → 프로파일이 **실행된 모든 커널을 빠짐없이 기록**함이 증명됨. 따라서 §1·§4b의 ORT per-op 수치는 전부 **post-optimization 실측**.

### INT8 lowering: 저장 그래프(onnx) → 디바이스 실제 실행(프로파일)
| op (저장) | 저장 개수 | → 디바이스 실행/iter | 무슨 일이 일어났나 |
|---|---|---|---|
| Conv | 113 | **QLinearConv 113** | 각 Conv + 주변 Q/DQ → 1개 QLinearConv로 융합 |
| QuantizeLinear | 415 | **4** | QDQ fusion이 ~411개를 QLinear*에 흡수 |
| DequantizeLinear | 643 | **5** | QDQ fusion이 ~638개 흡수 |
| Sigmoid / Mul | 102 / 103 | QLinearSigmoid 102 / QLinearMul 103 | SiLU |
| Concat | 26 | QLinearConcat 26 | |
| Add | 21 | QLinearAdd 21 | |
| MaxPool | 3 | **NhwcMaxPool 3** | NHWC 도메인으로 변환 |
| MatMul / Softmax | 2 / 2 | QLinearMatMul 2 / QLinearSoftmax 2 | attention |
| Transpose | 3 | **15** | NHWC 경계 레이아웃 변환 +12 삽입 |
| Split / Reshape / Resize | 10 / 11 / 2 | 11 / 11 / 2 | |
| **합계** | **1461 노드** | **423 실행커널/iter (16종)** | **Q/DQ 1058개(저장의 72%) → 9개로 붕괴** |

→ **저장 모델(1461)을 세면 3.5× 과대계상.** 실제 실행은 423커널/iter, 그중 **113 QLinearConv가 75.7% 시간** 차지. QDQ 스캐폴딩은 fusion으로 사실상 소멸.

### FP32 lowering
저장 403노드 → 실행 295커널/iter(16종). Sigmoid(102)+Mul(103) → **QuickGelu 101**로 융합. Conv 113 유지(NCHW, fp32는 NHWC 변환 안 함 §4 확인). 

### ★ 호스트(x86) 최적화는 디바이스(ARM)와 다르다 — 반드시 디바이스 기준
`SetOptimizedModelFilePath`로 호스트 x86에서 최적화하면 **NchwcTransformer(x86 전용 blocked NCHW)**가 돌아 ReorderInput/Output 노드가 생기고 int8 fusion도 부분적(QLinearConv 10개만, Q/DQ 1089개 잔존). ARM 디바이스는 **NhwcTransformer**로 113 QLinearConv 완전 융합. ORT 경고도 "hardware specific, 같은 환경에서만"이라 명시. → **op 개수/병목은 디바이스 런타임 프로파일로만 측정해야 정확**(우리가 한 방식).

### 우리 vs ORT INT8 — op 개수는 거의 동일, 차이는 op당 속도뿐
| 기능 | 우리(layer) | ORT INT8(실행) | 비고 |
|---|---|---|---|
| conv | conv2d 112 | QLinearConv 113 | **개수 일치** |
| concat | 26 | QLinearConcat 26 | 일치 |
| pool | 3 | NhwcMaxPool 3 | 일치 |
| upsample | 2 | Resize 2 | 일치 |
| split/slice | slice 22 | Split 11 | 우린 split을 2 slice로 |
| add | 19 | QLinearAdd 21 | 거의 일치 |
| attention | psa_attention 1 | QLinearMatMul 2+QLinearSoftmax 2 | 우린 1 레이어로 융합 |
| SiLU | (conv에 융합) | QLinearSigmoid 102+QLinearMul 103 | 우린 conv epilogue |

→ **op 구성/개수는 사실상 동일.** 7× 격차는 op 개수가 아니라 **op당 실행 속도**(conv 5.5×, slice 171×, …)에서 전적으로 나온다. = 알고리즘/그래프 구조 문제 아님, **커널·데이터플로우 문제** 확정.

---

## 5. 산출물 위치
- ONNX 모델/하니스: `artifacts/onnx/` (fp32 80MB, int8 21MB, export_onnx.py)
- ORT 디바이스 하니스: scratchpad `ort_android/ort_bench.cpp` + AAR 1.23.2
- ORT 프로파일 JSON: scratchpad `prof_int8.json`(4.4MB) / `prof_fp32.json`(2.9MB)
- 마스터플랜: `artifacts/yolov11_w4a8_masterplan.md`
