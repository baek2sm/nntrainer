# YOLOv11m W4A8 — Baseline v27 종합 기록 & 최적화 플래닝

Updated: 2026-07-02
Commit: `b01bfb3c` (backup branch `yolov11-device-opt-baselinev27`)
Branch: `myopt/w4a8-static-q8`
Device: SM-S938N / Snapdragon 8 Elite (serial R3CY10WM83Y), Android, thread=8
Spec: `~/projects/docs/nntrainer/yolov11m_w4a8_static_q8_spec.md`

이 문서는 v27 시점의 **모든 재현 정보 + 정직한 현재 상태 + 앞으로의 방향**을 한 파일에
담는다. 대화 히스토리는 휘발하므로 이 문서 + `task.md` + `handoff.md`가 유일한 생존자다.

---

## 0. 한눈에 (TL;DR)

| 항목 | v27 실측값 (device, 2026-07-02) | 비고 |
|---|---|---|
| e2e latency | **354.263 ms** (avg / 5 iters) | census off = v27과 수치 동일 |
| Peak RSS | **429,900 kB (~420 MB)** | 목표 <300MB 미달 (NHWC 기존이슈) |
| 검출 | 2개, conf **0.927363 / 0.887984** | 런 간 bit-identical |
| int8 activation 엣지 | **3 / 288** (fp16 285) | ★핵심 진실 — 아래 §3 |
| gate R (unittest_layers) | 1031 / 1031 PASS | build_x86 cwd |

**요지:** v27은 "W4A8 인프라가 코드로 존재하고, conv 1×1 3엣지만 실제 int8로 도는" 상태다.
속도(354ms)가 baseline(~352ms)에서 안 준 것은 **정상** — 288개 엣지 중 3개만 int8이라 이득이
날 수가 없다. 다음 단계는 "모든 activation 엣지를 int8로 통일"(fp16 285→0)하여 진짜 int8
baseline을 확보하고, 그 위에서 conv 최적화를 재개하는 것.

---

## 1. v27가 무엇인가 (커밋 구성)

HEAD `b01bfb3c` 위 4개 클린 커밋이 v27의 W4A8 정적 Q8_0 활성 경로를 구성한다:

```
b01bfb3c [Application] YOLOv11 W4A8 static Q8_0 activation scale injection
e9cb6cc6 [graph]       Promote calibrated conv edges to persistent int8 activation
ebd838b0 [layers]      Add int8-native static Q8_0 activation path to Conv2D
79b4b9f4 [tensor]      Plan backing memory for FORWARD_INFER-lifespan tensors
```

- **79b4b9f4 [tensor]** — FORWARD_INFER 수명 텐서의 backing memory를 메모리 플래너가
  잡도록. int8 활성 scratch 사전할당 기반.
- **ebd838b0 [layers]** — Conv2D에 int8-native 정적 Q8_0 경로 추가. 입력 1×1 int8 소비
  (`pack_int8_nhwc_q8_0x4_rows`+`Q8_0_Tensor::dot_prepacked_x4`), 출력 epilogue
  requant(`requant_q8_0_tw_from_fp16`), 캐패빌리티 게이트
  (`supportInt8ActInput/Output`), props(`ActivationScale`/`InputActivationScale`/`PreactScale`).
- **e9cb6cc6 [graph]** — 캘리브된 conv 엣지를 영속 int8 활성으로 승격. §5.7 propagation
  (`propagateActivationDataTypes`, 보수적 3조건), out_dim Q8_0_TW 타이핑
  (network_graph.cpp:1018/1200).
- **b01bfb3c [Application]** — YOLOv11 앱에서 W4A8 정적 Q8_0 scale 주입(calib json →
  그래프 메타). `w4a8`/`w4a16` 프리셋, `NNTR_CONV_Q8ACT` 처리.

**Q8_0_TW (tensor-wise static Q8_0):** flat int8 payload (1 byte/element), per-tensor
scale은 그래프 메타데이터로만 보유(텐서에 block scale 없음). `DataType` enum = 14.

---

## 2. 재현 방법 (device)

### 2.1 빌드 (lib 전용, ABI 불변)
```bash
export ANDROID_NDK=/tmp/android-ndk-r26d
export PATH=$ANDROID_NDK:$PATH
ninja -C build_android          # rc=0 확인
```

### 2.2 배포
```bash
DEV=/data/local/tmp/nntrainer
adb push build_android/jni/arm64-v8a/libnntrainer.so $DEV/
```

### 2.3 실행 (v27 정확성/속도/메모리 실측 — census OFF)
```bash
DEV=/data/local/tmp/nntrainer
adb shell "cd $DEV && LD_LIBRARY_PATH=$DEV \
  OMP_NUM_THREADS=8 NNTR_NUM_THREADS=8 \
  YOLO_TENSOR_TYPE=w4a8 \
  YOLO_BENCH_ITERS=5 \
  YOLO_ACT_SCALES=res/yolov11m_calib10.act_scales.json \
  YOLO_WEIGHTS=yolov11m_fw_q40_arm.safetensors \
  ./yolov11_infer $DEV/res $DEV/res/input_cat.bin 2>&1"
```
기대 출력: `Inference time: ~354 ms` / `Peak RSS: ~429900 kB` /
검출 2개 `conf 0.927363, 0.887984 (cls 0)`.

### 2.4 엣지 센서스 (정직 게이트 — census ON)
```bash
adb shell "cd $DEV && LD_LIBRARY_PATH=$DEV OMP_NUM_THREADS=8 \
  NNTR_EDGE_CENSUS=1 \
  YOLO_TENSOR_TYPE=w4a8 YOLO_BENCH_ITERS=1 \
  YOLO_ACT_SCALES=res/yolov11m_calib10.act_scales.json \
  YOLO_WEIGHTS=yolov11m_fw_q40_arm.safetensors \
  ./yolov11_infer $DEV/res $DEV/res/input_cat.bin 2>&1" | grep EDGECENSUS
```
기대: `[EDGECENSUS] total=288 int8=3 fp16=285 | blockers: no-emit=231 no-scale=0 consumer=54`
(`NNTR_EDGE_CENSUS`는 network_graph.cpp의 **임시 계기** — PR 전 제거 대상.)

### 2.5 gate R (수치 회귀 게이트)
```bash
cd build_x86 && ./test/unittest/unittest_layers   # 1031/1031 PASS
```

### 2.6 입력 자산 (device `/data/local/tmp/nntrainer/res/`)
- `yolov11m_fw_q40_arm.safetensors` — Q4_0(4-bit) 가중치, ARM repack 레이아웃.
- `yolov11m_calib10.act_scales.json` — 정적 캘리브레이션 활성 scale (564 엔트리).
  로컬 원본: `.claude/orchestrator/tasks/yolov11-w4a8-static-scale/artifacts/gate_a_calib10/yolov11m_calib10.act_scales.json`
  키 형식: `<node>`(출력 scale), `<node>:in`(입력 scale), `<node>:preact`,
  일부 `/generated_out_0`. 엣지 scale 정합(producer_out == consumer_in) 확인됨.
- `input_cat.bin` — 고양이 2마리 테스트 입력 (640×640).

---

## 3. ★핵심 진실 — 엣지 센서스 (정직 계기)

이전 세션에서 "activation 8bit end-to-end 됐다 / q8_0 통일 achieved"라는 **거짓 완료
보고를 반복**했다. 실제로는 int8 엣지가 3개뿐이었다. 이를 막기 위해 **device가 직접 찍는
객관 수치**(NNTR_EDGE_CENSUS)를 계기로 만들었다.

```
[EDGECENSUS] total=288 int8=3 fp16=285 | blockers: no-emit=231 no-scale=0 consumer=54
```

- **int8=3** — 실제 int8 엣지는 `m8/cv2`, `m9/cv2`, `m10/ffn0` 3개뿐(모두 1×1 conv→conv).
- **fp16=285** — 나머지 전부 FP16 활성 엣지. **완료 게이트 = fp16을 0으로 만드는 것.**
- blockers:
  - `no-emit=231` — 생산자가 int8 출력을 거부(대부분 비-conv 레이어 + 1×1 아닌 conv).
  - `consumer=54` — 생산자는 int8 가능하나 소비자 중 하나가 int8 입력 거부.
  - `no-scale=0` — **오해 주의**: 이 값이 0인 것은 scale이 다 있어서가 아니라, blocker
    체크가 `!emit`에서 먼저 단락되어 대부분 scale까지 도달하지 않기 때문. (calib은 별도로
    전 엣지 커버 확인됨 — §2.6.)

**완료 판정은 내 말이 아니라 이 로그 숫자다.** fp16=0 + 검출 보존이 유일한 "됐다"의 근거.

---

## 4. 깨달은 점 (realizations)

1. **"인프라 있음" ≠ "동작함".** int8 conv 커널(1×1·3×3), epilogue requant, props,
   propagation이 전부 코드로 존재하지만 캐패빌리티 게이트가 `k==1 && s==1 && f%32==0`
   (1×1 전용)으로 잠겨 있어 3엣지만 활성. 인프라 존재를 완료로 착각하면 안 된다.

2. **속도가 안 준 것은 버그가 아니라 산수.** 288 중 3 엣지 int8 → 대부분 경로는 여전히
   FP16 dequant/requant를 왕복. 진짜 이득은 대다수 엣지가 int8로 통일된 뒤에야 나온다.

3. **calib은 이미 충분하다.** 564 엔트리가 conv뿐 아니라
   slice/cat/pool/add/attn/dw/upsample/res/vcat/proj/qkv/pe까지 전 엣지의 scale을 제공.
   즉 통일을 막는 것은 데이터가 아니라 **레이어의 int8 캐패빌리티 부재**다.

4. **아키텍처는 "레이어별 공용헬퍼"가 맞다(사용자 승인).** 프레임워크 일반경계
   (`LayerNode::forwarding`에서 자동 dequant/requant)는 공유 컨텍스트·in-place·backward에
   blast radius가 커 위험. 대신 각 비-int8 레이어가 공용헬퍼 2줄(dequant_in/requant_out)을
   호출 → 레이어별 독립 소유 → **분업 가능** + 센서스가 레이어 전환마다 점진적으로 감소.

5. **detection head는 int8 불가 구간이 있다.** cv3_2(80 클래스)처럼 out_ch가 32의 배수가
   아니면 필터가 FP로 남아 int8 출력을 정직하게 거부해야 한다(억지 int8 금지).

6. **RSS 420MB는 NHWC 기존 이슈.** w4a16도 동일 → int8 통일과 별개의 레버.

---

## 5. 해야 할 것 (TODO — int8 통일 서브태스크)

목표: 센서스 **fp16 285 → 0**, 검출 보존, 회귀 없음. 그 지점을 v28로 태깅.

| # | Subtask | 파일 범위 | 상태 |
|---|---|---|---|
| i0 | 엣지 센서스 계기 (정직 게이트) | network_graph.cpp | ✅ (device 3/288) |
| i1 | 공용헬퍼 헤더 (dequant Q8_0_TW→FP / requant FP→Q8_0_TW) | act_int8_boundary.h (신규) | ⚪ |
| i2 | conv 3×3 int8-native 확장 + 전-conv emit | conv2d_layer.cpp | ⚪ |
| i3 | propagation 정책에 확장 캐패빌리티 반영 | network_graph.cpp | ⚪ |
| i4 | app scale 주입 비-conv 커버 확인/확장 | Applications/YOLOv11 | ⚪ |
| i5+ | 플럼빙 레이어별 dequant/requant (slice/cat/add/pool/upsample/attn/dw) | 각 layer.cpp | ⚪ |
| iZ | 센서스 fp16=0 + 검출보존 확인 → **v28 태깅** | - | ⚪ |

**다음 액션:** U-i1 공용헬퍼 헤더 작성
(conv의 `requant_q8_0_tw_from_fp16`(conv2d_layer.cpp:550-558) hoist + dequant 신규) →
U-i2 conv 3×3 int8-native(`__ggml_q4_0_4x8_q8_0_indirect_GEMM_q8_0` +
`gather_conv_act_rows_q8_0` 이미 존재) 확장 → 센서스 재측정으로 감소 확인.

### 5.1 PR 전 반드시 제거할 임시 계기
- `network_graph.cpp` — `NNTR_EDGE_CENSUS` 블록.
- `ggml_interface_fp16.cpp` — `ConvKernelProf`/`CONVPROF` (`NNTR_PROFILE_CONV`).
- `neuralnet.cpp` — `OpProf` (`NNTR_PROFILE_OPS`).
  (현재 워킹트리에 uncommitted 상태 — 이 문서 커밋에는 미포함.)

---

## 6. 최종 최적화 목적지 (방향 플래닝)

```
[현재 v27]  354ms / 420MB / int8=3
     │  ── int8 통일 (i1..iZ) ──  ★분업 가능★
     ▼
[v28]  fp16=0 · 검출 보존 = "진짜 int8 baseline"
     │  여기서부터 속도 레버가 유효해진다:
     │   · conv int8 GEMM 최적화 (i8mm/SMMLA 활용, dequant 왕복 소거)
     │   · NHWC 데이터플로우 정리 → RSS <300MB
     │   · 비-conv 플럼빙 레이어 int8 경량화 (분업)
     ▼
[목표]  intermediate <200ms  →  최종 <100ms, RSS <300MB, 정확도 보존
        (ORT INT8 parity ~208-247ms 를 넘어서는 지점)
```

**핵심 순서 원칙:** 최적화보다 **통일이 먼저**다. 3엣지 위에서 conv를 최적화하면 국소
이득이 dequant/requant 왕복에 먹혀 측정이 오염된다. 전 엣지 int8(v28)을 확보해야
conv 최적화의 실제 이득이 latency에 그대로 드러난다. 통일 완료 후 conv가 최적임을 확인하고,
남은 레이어별 int8 최적화는 다른 담당자와 분업한다(공용헬퍼 아키텍처가 이를 가능케 함).

---

## 7. 표준 제약 (항상 유효)

- 커밋 author/Signed-off-by = `sb92.hong@samsung.com`만 (`ax.samsung.com` 커밋 금지).
- SiLU LUT 스킵 / lazy Q4_0 경로 금지 / framework-first(raw buffer bypass 금지).
- adb 세션 동시 2개 금지(OOM) / `subprojects/` 편집 금지 / 플랫폼 가드 필수.
- PR 전 임시 계기 전부 제거 / 검증된 baseline마다 `yolov11-device-opt-baselinevN` 백업.
- 한 커밋 = 한 주제 / 하드코딩 pass·허용오차 완화 금지.
