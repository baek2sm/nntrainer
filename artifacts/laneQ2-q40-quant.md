# Q4_0 offline quantization — artifacts/laneQ2-q40-quant.md

## 변경 파일
- **신규**: `Applications/CausalLM/models/YOLOv11/PyTorch/quantize_q4_0_conv.py`
- **생성물**: `Applications/CausalLM/models/YOLOv11/res/yolov11m_fused_q40.safetensors`

---

## 기존 상태 (baseline)

`yolov11m_fused.safetensors` (FP32, 76.4 MB)는 모든 conv filter를 FP32로 저장.
1x1 conv weight는 표준 `[out_ch, in_ch, 1, 1]` 4D 레이아웃.
기존 `quantize_int4_conv.py`는 QINT4 per-channel affine 포맷 — 이번 작업과 무관.

`conv2d_layer.cpp` finalize (line 342-351):
- Q4_0 dtype + groups==1 + kh==kw==1 이면 `quant_matmul_filter=true`
- filter `TensorDim = [1, 1, in_ch, out_ch]` (K=height=in_ch, N=width=out_ch)

`layer_devel.h` save path (line 403-412):
- `weight.transpose("0:2:1")` → [1,1,N=out_ch,K=in_ch]
- `quantize_q4_0(transposed, nrow=N, n_per_row=K)` → 평탄 block_q4_0 stream
- `repack_q4_0(plain, N, K, target_isa)` → q4_0x4(ARM) / q4_0x8(x86)
- ARM device: `__ggml_repack_q4_0_to_q4_0_4`; XOR mask 0x88 적용

Load path (`Q4_0_Tensor::read`):
- `getMemoryBytes() = size() = N*K/32 * 18`를 파일에서 직접 읽음
- **load 시 repack 없음** — 파일에 이미 repacked 포맷이 있어야 함

---

## 구현 내용

### quantize_q4_0_conv.py

**포맷 (ggml Q4_0, `quantize_row_q4_0_ref` 완전 일치)**:
- 블록 32원소: `d = max(x) / -8.0` (부호 보존, max는 최대 절댓값을 가진 값)
- `qi = clip(round(x / d) + 8, 0, 15)`
- nibble packing: `qs[j] = qi[j] | (qi[j+16] << 4)` for j in [0,16)
- d → fp16 little-endian 2바이트, qs 16바이트 → 블록당 18바이트

**전치 전략**:
- 입력 `[out_ch, in_ch, 1, 1]` → squeeze to `[out_ch=N, in_ch=K]`
- `quantize_q4_0(w2d)`: N rows × K cols (= 런타임 transpose + quantize와 동일)

**repack (q4_0x4, ARM)**:
- `repack_q4_0(raw, N=out_ch, K=in_ch, interleave=4)`
- 그룹당 4개 row: d[4]×2바이트 + qs[64]바이트 (XOR 0x88 적용)
- 총 바이트 = N×K/32×18 (plain과 동일 크기, 인-플레이스 재배치)

**safetensors 헤더**:
- `dtype: "U8"` (불투명 blob)
- `nntr_dtype: "Q4_0"` (로드 시 Q4_0_Tensor 선택)
- `nntr_shape: [1, 1, K=in_ch, N=out_ch]` (Q4_0_Tensor dim 요건: batch=1, channel=1, width%32=0)
- `shape: [blob_bytes]` (flat byte count)

**제외 규칙**:
- kh!=1 || kw!=1 → FP32 유지
- `dw:filter` (depthwise) → FP32 유지
- `out_ch % 32 != 0 || in_ch % 32 != 0` → FP32 유지 (block alignment)

**`--target` 옵션**:
- `arm` (default): q4_0x4 (device, S23)
- `x86`: q4_0x8 (desktop/CI)

---

## 결과

### 양자화 통계

| 항목 | 수 |
|------|-----|
| Q4_0 양자화된 1×1 conv | **57개** |
| 제외 (alignment: out_ch=1 not div32) | 3개 (det0/1/2 cv3_2 = 최종 검출 헤드 출력) |
| FP32 유지 (3×3 conv, bias, depthwise 등) | 164개 |

### 바이트 절감 (1×1 conv 가중치만)

| | 크기 |
|-|------|
| 1×1 conv FP32 합계 | 30,097,408 bytes (28.7 MB) |
| 1×1 conv Q4_0 합계 | 4,232,448 bytes (4.0 MB) |
| 압축률 | **14.1%** (= 87% 절감, 이론 FP32→Q4_0: 32/4 = 8×) |

### 파일 크기

| 파일 | 크기 |
|------|------|
| `yolov11m_fused.safetensors` (FP32) | 76.4 MB |
| `yolov11m_fused_q40.safetensors` (Q4_0) | **51.8 MB** |

총 파일 절감 −24.6 MB (−32%). 나머지(3×3 conv, bias, BN 파라미터)는 FP32 유지.

### 정확성 검증 (m6/cv1 [512, 512, 1, 1])

- d 값 일치: `d_quantized=-0.016251 == d_expected=-0.016251` (fp16 반올림 포함)
- 최대 양자화 오차: 0.0346 (Q4_0 4bit 수준, 정상)
- 평균 절대오차: 0.00292
- 상대 최대오차 1.0: 0에 가까운 가중치에서 발생하는 정상 현상

### Blob 레이아웃 요약

```
[U8 blob, N*K/32 * 18 bytes]
  q4_0x4 (ARM): 그룹 4행 × (K/32) 슈퍼블록
  각 슈퍼블록 = d[4]×uint16 (8바이트) + qs[64바이트] (XOR 0x88)
  nntr_shape = [1, 1, K=in_ch, N=out_ch]
```

---

## 인수기준 충족 여부

| 인수기준 | 충족 |
|----------|------|
| `quantize_q4_0_conv.py` 신규 생성 (int4 버전 기반) | ✓ |
| ggml Q4_0 블록 포맷 정확 (d=max/-8, nibble 패킹) | ✓ (`quantize_row_q4_0_ref` 와 d 값 일치 확인) |
| 전치 `[out_ch,in_ch] → [N,K]` 방향 정확 | ✓ (런타임 layer_devel.h transpose 경로와 동일) |
| in_ch%32 || out_ch%32 인 conv 제외 보고 | ✓ (3개: det0/1/2 cv3_2, out_ch=1) |
| ARM q4_0x4 repack + XOR 0x88 | ✓ (gguf_to_nntrainer.py 동일 구현, 독립 검증) |
| safetensors nntr_dtype="Q4_0", nntr_shape=[1,1,K,N] | ✓ |
| `yolov11m_fused_q40.safetensors` 생성 | ✓ (51.8 MB) |
| `subprojects/` 미편집 | ✓ |
| `conv2d_layer.cpp` 미편집 | ✓ |

---

## 남은 TODO / 가정

1. **런타임 연결 (PM 범위)**: `conv2d_layer.cpp`의 `quant_matmul_filter` 분기가 `Q4_0` dtype일 때 실제 발동하도록 YOLOv11 모델 설정 파일(`.ini` 또는 앱 코드)에서 weight dtype을 Q4_0으로 지정해야 함.

2. **C2PSA 레이어의 1×1 conv**: `m10/qkv`, `m10/proj`, `m10/ffn0`, `m10/ffn1` 포함 총 4개 양자화됨. C2PSA는 커스텀 레이어이므로 해당 레이어 코드에서 Q4_0 weight를 올바르게 소비하는지 확인 필요.

3. **정밀도 영향**: Q4_0 는 4bit 블록 양자화로 약간의 mAP 저하 예상. 실측 필요.

4. **x86 CI 빌드**: 현재 `--target arm` (q4_0x4). CI가 x86이면 `--target x86`으로 재생성 필요. 런타임에서 ISA 자동감지 후 load 시 repack 없이 사용하므로 타겟 일치 필수.
