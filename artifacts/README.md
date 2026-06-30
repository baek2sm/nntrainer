# YOLOv11m 디바이스 추론 최적화 — 진입점 (README)

> 이 폴더는 nntrainer의 YOLOv11m Android 추론을 ONNX Runtime(ORT) INT8 수준으로
> 끌어올리기 위한 **측정·분석·계획** 산출물이다. 처음 보는 사람이 배경부터 결론·계획까지
> 순서대로 따라올 수 있게 구성했다. 모든 수치는 **동일 기기·동일 입력 실측**이다.

---

## 0. 한 문단 요약 (지금 어디에 있나)

우리 YOLOv11m(832×832, nc=1)은 같은 기기에서 **1210.9ms/iter**, ORT INT8은 **208ms** = **5.8× 느림**.
정밀 프로파일 결과, **연산 개수도 연산량(MAC 57.2G)도 ORT와 동급**이며, 격차는 100%
**op당 실행 속도**(커널 효율 + 데이터플로우)에서 나온다 — 알고리즘/그래프 문제가 아니다.
격차는 두 부류다: **(a) conv 5.5×**(어려운 본진, NHWC+W4A8+SMMLA 필요)와
**(b) 비-conv 스칼라 디스패치 ~287ms**(독립적·쉬움). 동급 도달 계획은
`yolov11_parity_masterplan.md`에 4단계로 정리(P0 위생→P1 W4A8→P2 NHWC+SMMLA→P3 초과).
**상태: 측정/분석/계획 완료, 구현 미착수.**

---

## 1. 배경 — 무엇을, 왜 비교하나

- **모델**: YOLOv11m, 입력 832×832×3, 단일 클래스(nc=1, person). 약 57.2 GMAC/iter.
- **우리 구성**: nntrainer, NCHW, **Q4_0 가중치 + FP16 활성**(현재 디바이스 러너). ThreadManager
  기반(OpenBLAS 미사용), 4 compute thread. 백엔드 conv = ggml i8mm indirect GEMM(**SMMLA**, LLM prefill FC와 동일 커널; GEMM 명령 자체는 이미 ORT급 ~430 GMAC/s).
- **비교 대상**: ONNX Runtime 1.23.2, 같은 모델을 export 후 두 가지로 실행.
  - **ORT FP32** (NCHW) — "양자화 없이도 우리가 느린가?"의 apples-to-apples 기준.
  - **ORT INT8** (NHWC, per-channel weight / per-tensor activation) — **최종 목표 수준**.
- **왜 ORT INT8가 목표인가**: 같은 기기에서 실측 208ms. 모바일 추론 SOTA 레퍼런스이고,
  우리가 도달해야 할 "동급" 정의의 기준점.
- **기기**: Samsung **SM-S926U (Galaxy S24+)**, Snapdragon **8 Gen 3 / SM8650 (pineapple)**.
  SMMLA(i8mm, ARMv8.6)·SDOT(ARMv8.2) 모두 지원. 측정 시 serial 예: `R3CW808LKAE`(동일 모델).
- **입력**: `input_cat.bin` = [1,3,832,832] float32 실사진(고양이 2마리). 검출 기준값
  0.9258 / 0.8868(두 박스 conf) — 정확성 회귀 게이트로 사용.

### 핵심 사실(직관과 다른 것들)
1. **우리 FP32-FP32(3130ms)가 ORT FP32(841ms)보다 3.7× 느리다** → 양자화를 아예 안 해도
   느림 = 격차는 양자화 부족이 아니다. 근본 원인은 우리 FP32 GEMM이 naive `__fallback_sgemm`.
2. **NHWC는 INT8에 결합된 레버다.** ORT조차 FP32는 NCHW로 둔다. NHWC는 int8 SMMLA를
   풀로 먹이기 위한 전제이지, FP16/FP32에서 단독으로 큰 레버가 아니다.
3. **op 개수/MAC량은 ORT와 동급.** 7× 격차는 전적으로 op당 속도 문제.

---

## 2. 문서 지도 (읽는 순서)

| 순서 | 문서 | 내용 |
|---|---|---|
| ① | **README.md** (이 문서) | 배경·재현·방법론·현황 진입점 |
| ② | **yolov11_device_gap_analysis.md** | 핵심 분석. §0 e2e 실측 → §1 ORT per-op → §2 ORT가 빠른 기제 → §3 우리가 느린 기제 → §4b 우리 전체 디테일 분해(표) → §4c ONNX lowering 검증 |
| ③ | **yolov11_parity_masterplan.md** | 동급 도달 계획(측정 근거·projection·리스크·검증규율). 트랙 분담(우리=conv) |
| ③' | **yolov11_conv_workplan.md** | ★위임용 작업계획서. conv 트랙을 최소 독립단위(A~E)로 쪼갬 — 각 단위 진입점·구현지침·**정량 인수기준(G-BIT/G-TOL/G-UNIT)**·측정법 자기완결. 외부 위임→결과 회수용 |
| ③'' | **DEVICE_MEASURE_PROTOCOL.md** | ★3-에이전트 공유 디바이스 측정 조율. 단일 ADB 기기를 락(`projects/device_lock/`)으로 직렬화. 측정 전 acquire/후 release, BUSY면 비-device 작업 먼저 후 재시도(스킵 금지) |
| 보조 | yolov11_w4a8_masterplan.md | ②③ 이전의 W4A8 심층 분석(lever #1 배경). ③에 통합·갱신됨 — 역사적 맥락용 |
| 보조 | onnx/EXPORT_NOTES.md, export_onnx.py | ONNX export/quantize 재현 |
| 보조 | laneQ2/laneP5, fp16_integration_status | 과거 단계별 구현 기록 |

---

## 3. 재현 레시피 (어떻게 이 숫자들을 얻었나)

### 3.1 우리 측정 (디바이스)
```bash
# 1) 라이브러리 빌드 (meson가 ndk-build 래핑; 증분 ~3s)
ninja -C build_android
# 2) 디바이스로 새 lib 푸시 (앱은 .so를 런타임 링크 → 앱 재빌드 불필요)
adb -s <serial> push build_android/jni/arm64-v8a/libnntrainer.so /data/local/tmp/yolov11/
# 3) 실행 (3개 프로파일러 동시 ON, 8 iter 평균)
adb -s <serial> shell 'cd /data/local/tmp/yolov11 && LD_LIBRARY_PATH=. \
  YOLO_TENSOR_TYPE=FP32-FP16 YOLO_CONV_Q40=1 \
  YOLO_WEIGHTS=yolov11m_fw_q40_arm.safetensors \
  YOLO_BENCH_ITERS=8 \
  YOLO_LAYER_PROFILE=1 YOLO_CONV_GEOM=1 YOLO_KERNEL_PROFILE=1 \
  ./yolov11_infer . input_cat.bin'
```
디바이스 파일(`/data/local/tmp/yolov11/`): `yolov11m_fw_q40_arm.safetensors`(Q4_0 weight),
`input_cat.bin`, `yolov11_infer`(앱), `libnntrainer.so`/`libccapi-nntrainer.so`/`libc++_shared.so`.
원시 출력: `artifacts/raw/our_profile_8iter.txt`.

### 3.2 ORT 측정 (디바이스)
```bash
# 하니스: scratchpad ort_android/ort_bench.cpp (AAR 1.23.2, 4 intra-op thread, ORT_ENABLE_ALL)
adb -s <serial> push ort_bench libonnxruntime.so libc++_shared.so *.onnx input_cat.bin /data/local/tmp/ort/
adb -s <serial> shell 'cd /data/local/tmp/ort && LD_LIBRARY_PATH=. \
  ./ort_bench yolov11m_832_int8.onnx input_cat.bin 10 prof_int8'   # FP32: yolov11m_832_fp32.onnx
```
프로파일 JSON: `prof_int8.json`(4.4MB)/`prof_fp32.json`(2.9MB). 집계 스크립트는 §4 참고.

### 3.3 ONNX export/quantize
`artifacts/onnx/export_onnx.py` (+ EXPORT_NOTES.md). 산출: `yolov11m_832_fp32.onnx`(80MB),
`yolov11m_832_int8.onnx`(21MB, static per-channel).

---

## 4. 방법론 — 계측이 신뢰할 만한 이유

### 4.1 우리 쪽 3중 계측 (env로만 ON, 평소 오버헤드 0; **작업트리 변경, 미커밋**)
| env | 파일 | 측정 |
|---|---|---|
| `YOLO_LAYER_PROFILE` | `nntrainer/models/neuralnet.cpp` (LayerProf) | 레이어 타입별/이름별 forward 시간 + 개수 |
| `YOLO_CONV_GEOM` | `nntrainer/layers/conv2d_layer.cpp` (ConvGeomProf) | conv 카테고리별(커널크기·종류·dtype·경로) 시간+개수+**이론 MAC→GMAC/s** |
| `YOLO_KERNEL_PROFILE` | `.../ggml_interface/ggml_interface_fp16.cpp` (ConvKernelProf) | 3×3 indirect 커널 내부: gather+재양자화 vs GEMM 분리 |

- **MAC 검증**: 계측이 합산한 총 MAC = 57.2 GMAC/iter = YOLOv11m@832 이론치와 일치 → 계측 신뢰.
- **GMAC/s**가 핵심 지표: 동일 MAC을 누가 더 빨리 처리하나. 우리 66 vs ORT 365 = 5.5×.

### 4.2 ORT 쪽 — 저장 그래프 ≠ 실제 실행 (lowering 검증)
- ORT는 그래프 최적화(QDQ fusion, NHWC 변환)를 하므로 **저장 모델 op 개수 ≠ 실행 개수**.
  저장 INT8=1461노드지만 디바이스 실제 실행=**423커널/iter**(Q/DQ 1058개→9개 붕괴).
- **프로파일 완전성 검증**: node 시간 합 / model_run = **98.7%(INT8)/99.7%(FP32)** →
  실행 커널 누락 0 → per-op 수치는 전부 post-optimization 실측.
- **호스트(x86) 최적화는 디바이스(ARM)와 다름**(x86=NchwcTransformer, ARM=NhwcTransformer).
  → op 개수/병목은 **디바이스 런타임 프로파일로만** 측정(이 방식 사용). 상세 §4c.

---

## 5. 핵심 결론 (납득 체인)

1. **느리다** — 1210.9ms vs ORT INT8 208ms = 5.8×. (§0)
2. **양자화 핑계 아님** — FP32끼리 비교해도 3.7× 느림. 우리 FP32 GEMM이 naive 폴백. (§0)
3. **op 개수·MAC량은 동급** — conv 112≈113, MAC 57.2G 동일. 그래프/알고리즘 문제 아님. (§4c)
4. **격차는 op당 속도** — conv 66 vs 365 GMAC/s(5.5×), slice 171×, pool 74×, upsample 9.4×. (§4b)
5. **두 부류로 분해** — (a) conv = W4A8+NHWC(어려움; **SMMLA는 이미 사용 중**, 격차는 NCHW 출력 transpose 142ms+재양자화 132ms 같은 dataflow), (b) 비-conv 287ms = 벡터화/memcpy(쉬움). (§4b·§3 정정)
6. **이미 이상적** — concat(우리 승), addition(동급)은 건드릴 필요 없음. (§4b)

---

## 6. 계획 & 현황

`yolov11_parity_masterplan.md` 4단계. projection(per-iter):

| 단계 | e2e | vs 208 | 내용 |
|---|---|---|---|
| baseline | 1210.9 | 5.8× | 현재 |
| P0 위생 | ~916 | 4.4× | slice/pool/upsample/attention 벡터화 + conv0 스템 |
| P1 W4A8 | ~666 | 3.2× | 층간 int8 영속 + quantize-once + int8 SiLU |
| P2 NHWC | ~247 | 1.19× | NHWC로 출력 transpose 제거 + 이미 쓰는 SMMLA를 오버헤드 없이 풀가동 (**parity**) |
| P3 fuse+tune | ~175 | 0.84× | epilogue 융합 + 튜닝 + decode (**초과**) |

**현황**: 측정·분석·계획 완료. 구현 미착수. 추천 시작점 = **P0**(독립·리스크 0·빠른 win).
진행 추적: task #7(P0)·#8(P1)·#9(P2)·#10(P3).

**검증 규율(단계마다)**: baseline 브랜치 push → 동일기기 A/B → 검출 bit-identical(P0)
또는 허용오차+mAP(P1·P2) → RSS 기록 → 3중 계측으로 재분해.
