# Timm ViT Implementation - Patch Embed Only

## 구현 개요

이 폴더는 timm 라이브러리의 ViT(Vision Transformer) 모델과 호환되는 nntrainer 구현체입니다. 현재는 `patch_embed` (Conv2D) 부분만 구현되어 있습니다.

## 파일 구조

```
Applications/CausalLM/models/timm_vit/
├── timm_vit_transformer.h          # TimmViTTransformer 클래스 헤더
├── timm_vit_transformer.cpp        # TimmViTTransformer 클래스 구현
├── main_vit_patch_embed.cpp        # 테스트 메인 파일
├── meson.build                     # 빌드 설정
└── README.md                       # 이 파일
```

## 구현된 기능

### 1. Patch Embedding (Conv2D)
- **파일**: `timm_vit_transformer.cpp`
- **기능**: 이미지를 patch로 임베딩하는 Conv2D 레이어
- **설정**:
  - Input: `[batch, 3, 224, 224]` (RGB 이미지)
  - Kernel: `[16, 16]` (patch_size)
  - Stride: `[16, 16]`
  - Output: `[batch, 768, 14, 14]` (embed_dim=768, 14x14 patches)
  - Padding: "same" (nntrainer conv2d 설정)
  - Bias: 비활성화 (disable_bias=true)

### 2. Weights Converter
- **파일**: `App.../res/vit/timm_vit_base_patch16_siglip_224/patch_embed_converter.py`
- **기능**: timm safetensors 형식을 nntrainer binary 형식으로 변환
- **사용법**:
```bash
source /home/seungbaek/miniconda3/etc/profile.d/conda.sh
conda activate 26_2
cd Applications/CausalLM/res/vit/timm_vit_base_patch16_siglip_224
python patch_embed_converter.py
```
- **출력**: `nntr_vit_patch_embed_fp32.bin`

### 3. Test Script (Python)
- **파일**: `App.../res/vit/timm_vit_base_patch16_siglip_224/test_patch_embed.py`
- **기능**:
  - timm ViT patch_embed 실행
  - 참조 출력 생성 (NumPy 형식)
  - 통계 정보 출력
- **사용법**:
```bash
source /home/seungbaek/miniconda3/etc/profile.d/conda.sh
conda activate 26_2
cd Applications/CausalLM/res/vit/timm_vit_base_patch16_siglip_224
python test_patch_embed.py
```
- **출력**:
  - `patch_embed_output_ref.npy`: timm 참조 출력
  - `patch_embed_weight_pytorch.npy`: PyTorch 형식 weight

## 테스트 결과

### timm (PyTorch) Reference Output
```
Patch embed output shape: torch.Size([1, 768, 14, 14])
Patch embed output statistics:
  Mean: 0.240311
  Std: 0.127753
  Min: -2.112407
  Max: 2.700375

First 10 values of output[0, 0, 0, 0:10]:
  tensor([0.0177, 0.0331, 0.0484, 0.0638, 0.0789, 0.0943, 0.1098, 0.1251, 0.1406, 0.1555])
```

## 다음 단계

### 1. nntrainer 빌드 및 테스트

현재 `Transformer::model` 멤버변수가 `protected`로 설정되어 있어, 직접 접근이 불가능합니다. public getter 메서드를 추가하거나 테스트를 위한 공개 인터페이스가 필요합니다.

#### 해결 방법 A: public getter 추가 (추천)

`transformer.h`에 getter 추가:
```cpp
public:
  ml::train::Model* getModel() { return model.get(); }
```

그 후 `main_vit_patch_embed.cpp`에서:
```cpp
model.getModel()->setDataBuffer((void **)&input_data, (void **)&label_data, batch_size, 1);
model.getModel()->infer();
auto output = model.getModel()->getOutput(0);
```

#### 해결 방법 B: protected 접근 허용 (테스트용)

`main_vit_patch_embed.cpp`에서 friend class 사용 또는 상속 접근:

### 2. 빌드 및 실행

```bash
# 빌드
cd /home/seungbaek/projects/nntrainer/build
cmake .. -GNinja
ninja

# 실행 (빌드 완료 후)
./main_vit_patch_embed
```

### 3. 결과 비교

nntrainer 출력 생성 후:
```python
import numpy as np

# Load outputs
ref_output = np.load('patch_embed_output_ref.npy')
nntr_output = np.load('patch_embed_output_nntrainer.npy')

# Compare
print("Mean difference:", np.mean(np.abs(ref_output - nntr_output)))
print("Max difference:", np.max(np.abs(ref_output - nntr_output)))
print("Are they close?", np.allclose(ref_output, nntr_output, atol=1e-5))
```

### 4. 다음 레이어 구현 (나중에)

patch_embed 검증 완료 후 다음 레이어들을 순서대로 구현:
1. Positional Embedding
2. Class Token
3. Transformer Blocks (12 layers)
   - LayerNorm 1
   - Multi-Head Attention (QKV, O)
   - LayerNorm 2
   - MLP (FC1, FC2)
4. Final LayerNorm
5. Attention Pool (global_pool="map"인 경우)

## Weight 저장 순서 (nntrainer 형식)

현재 `patch_embed_converter.py`에서 다음 순서로 저장:
1. `patch_embed.proj.weight`: `[768, 3, 16, 16]` (Conv2D weight)
2. `patch_embed.proj.bias`: `[768]` (Conv2D bias, 있을 경우)

Note: Conv2D의 경우 nntrainer와 PyTorch의 weight 형식이 동일하므로 transpose 불필요

## 참고 자료

- timm ViT 코드: 제공된 PyTorch 구현체 참고
- YOLOv3 Conv2D: `Applications/YOLOv3/jni/main.cpp` (conv2d 레이어 설정 참고)
- 이미지 로딩: `Applications/CausalLM/llm_util.cpp` (`loadAndPreprocessImage` 함수)
- 다른 CausalLM 모델 구조: `Applications/CausalLM/models/gemma3/`

## 주의사항

1. Conv2D padding 설정은 `same`으로 설정되어 있으나, timm에서는 padding이 적용되지 않을 수 있음
2. 현재 `main_vit_patch_embed.cpp`에서 `model.model` 접근 문제 해결 필요
3. 빌드 시 `meson.build`에 timm_vit가 추가되어 있음 확인 필요
4. 이미지 전처리는 현재 normalize=false로 설정되어 있음 (loadAndPreprocessImage 호출)

## Authors

Seungbaek Hong <sb92.hong@samsung.com>

 날짜: 2026년 1월 28일
