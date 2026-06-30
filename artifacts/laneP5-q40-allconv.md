# laneP5-q40-allconv: Q4_0 All-Conv Quantization

## 변경 파일
- `Applications/CausalLM/models/YOLOv11/PyTorch/quantize_q4_0_conv.py` — 확장 (기존 1×1 로직 보존 + `--all` 옵션 추가)
- `Applications/CausalLM/models/YOLOv11/res/yolov11m_fused_q40_all.safetensors` — 신규 생성 (산출물)

## 기존 상태 (baseline)
- `quantize_q4_0_conv.py`는 1×1 conv filter만 Q4_0 양자화 (57개). 3×3/stride2 등 larger kernel은 `[keep-ksize]`로 FP32 유지.
- `yolov11m_fused_q40.safetensors`: 57 Q4_0 + 167 FP32, 51.8 MB
- Q4_0 layout: filter `[out_ch, in_ch, 1, 1]` → squeeze `[N=out_ch, K=in_ch]` → quantize → repack → nntr_shape `[1,1,K,N]`

## 구현 내용

### 핵심 확장: `--all` 플래그
기존 코드는 `kh != 1 or kw != 1`이면 무조건 `[keep-ksize]`로 통과시켰다. `--all` 옵션 시 이 게이트를 제거하고 아래 일반화된 layout을 적용한다.

### Layout 일반화 (3×3 포함 모든 groups=1 conv)
```
FP32 filter in safetensors: [out_ch, in_ch, kh, kw]  (C-contiguous)
w.reshape(out_ch, CRS)  where CRS = in_ch * kh * kw
  -> K-index = ic*kh*kw + ki*kw + kj  (channel 바깥, kh, kw 안)
  -> nntrainer im2col 컬럼 순서와 동일 (conv2d_layer.cpp: c 루프 외→h→w 내)
w2d shape: [N=out_ch, K=CRS]  (quantize_q4_0가 기대하는 [N,K])
nntr_shape: [1, 1, CRS, out_ch]  (1×1과 동일 컨벤션)
```

1×1(kh=kw=1)이면 CRS=in_ch라 `w.reshape(out_ch, in_ch)` = 기존 `w[:,:,0,0]`과 동일 → 완전 호환.

### 제외 규칙 (FP32 유지)
- depthwise (name에 `dw:` 포함): `[keep-dw]`
- out_ch == 1 (degenerate): `[keep-excl]`
- CRS % 32 != 0 또는 out_ch % 32 != 0: `[skip-align]`
- out_ch % interleave != 0: `[skip-align]`
- `--all` 없이 kh*kw > 1: `[keep-ksize]`

### 리팩토링 포인트
- `is_eligible_1x1_conv_filter` → `check_conv_filter_eligibility(allow_larger_kernels=bool)` 로 대체
- `quantize_filter(raw, shape, interleave)` 헬퍼로 추출 (1×1/3×3 공통)
- 기존 `quantize_q4_0`, `repack_q4_0`, safetensors I/O 함수는 100% 동일 재사용

## 인수기준 충족 상황

| 기준 | 상태 |
|------|------|
| 1×1 + 3×3 + stride2 conv 전부 Q4_0 (`--all`) | ✓ 101개 (1×1: 57, larger: 44) |
| depthwise/out_ch=1/CRS%32≠0 제외 | ✓ depthwise 10개, out_ch=1 3개, CRS%32≠0 1개(conv0 CRS=27) |
| nntr_shape=[1,1,CRS,out_ch] | ✓ 모든 Q4_0 텐서 확인 |
| 기존 1×1 경로 회귀 없음 | ✓ md5sum byte-for-byte 동일 vs yolov11m_fused_q40.safetensors |
| 산출 파일 생성 | ✓ res/yolov11m_fused_q40_all.safetensors |

## 산출물 요약 (`--all --target arm`)

```
Input:  yolov11m_fused.safetensors      76.4 MB (FP32)
Output: yolov11m_fused_q40_all.safetensors  10.9 MB
  Q4_0 tensors: 101  (1x1: 57, larger: 44)
  Skipped (align): 1  -> conv0/conv:filter [64,3,3,3] CRS=27 not div32
  Kept FP32: 123  (bias, BN params, depthwise, out_ch=1)
  Quantized bytes: 79,937,536 FP32 -> 11,241,216 Q4_0 (14.1%)
```

샘플 3×3 항목 (nntr_shape 확인):
- `conv5/conv:filter [512,512,3,3]` → nntr_shape=[1,1,4608,512], 1,327,104 bytes ✓
- `m2/m0/inner0/cv1/conv:filter [32,32,3,3]` → nntr_shape=[1,1,288,32], 5,184 bytes ✓
- `det0/cv2_0/conv:filter [64,256,3,3]` → nntr_shape=[1,1,2304,64], 82,944 bytes ✓

## 주의 / TODO (런타임 PM 담당)
- 현재 런타임(`conv2d_layer.cpp`)의 `quant_matmul_filter`는 `kh==kw==1`에만 Q4_0 weight layout을 활성화한다.
  3×3 Q4_0 소비를 위해서는 런타임에서:
  1. `finalize()`: `quant_matmul_filter` 조건 확장 (`kh*kw>1` 포함) + filter dim을 `[1,1,CRS,out_ch]`로 설정
  2. `forwarding()`: Q4_0 3×3 path에서 `im2col(input)` → `[OH*OW, CRS]` 후 Q4_0 GEMM (`act[OH*OW,CRS] · weight[CRS,out_ch] → [OH*OW,out_ch]`) 구현 필요
  이 파일은 런타임이 준비되었을 때 바로 사용 가능한 포맷으로 사전 생성한 것이다.
