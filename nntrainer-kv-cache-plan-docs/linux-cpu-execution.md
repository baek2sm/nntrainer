# Linux CPU 개발·검증 기록

사용자 결정: **Linux x86 CPU → Android Arm CPU → 이후 backend는 추후 결정**.
`AGENTS.md`는 참고하며 수정하지 않는다. 구현·명령·검증·미완료 사항을 단계별로
이 문서에 갱신하고, 통과한 단위로 DCO/agent trailer를 갖춘 커밋을 남긴다.

## P0: 계획에 대한 비판적 검토

기준 소스는 계획과 같은 `141871a58ff439e3caeac40c1632ff8ac3543d20`이다.
따라서 소스 버전 차이보다 실행 환경과 모델 패키지의 차이를 먼저 해결한다.

| 계획의 가정 | 실제 관찰 / 적용 결정 |
|---|---|
| Ubuntu 24.04, GCC 13 | Ubuntu 22.04, GCC/G++ 11.4, Meson 0.61.2, Ninja 1.10.1, clang-format 14.0.0. 첫 기준은 비FP16 CPU build. native FP16 대신 wrapper도 별도 확인한다. |
| CPU 다음 OpenCL/QNN probe | 사용자 요청에 따라 Linux CPU를 먼저 완료하고 Android CPU로 진행한다. 장치 probe를 CPU 착수의 선행 조건으로 두지 않는다. |
| x86 tokenizer archive를 그대로 link | `esaxx.o`가 GCC 11 libstdc++에 없는 `std::ios_base_library_init()`을 요구한다. in-tree Rust crate를 현재 컴파일러로 별도 build하고 `causallm-tokenizer-lib`를 지정한다. 추적 archive를 변경하지 않는다. |
| 서브모듈 전체 초기화 | 이 Linux profile에 필요한 xgrammar(+하위 모듈), googletest를 고정 SHA로 초기화했다. OpenBLAS/iniparser는 시스템 라이브러리를 사용한다. vendored 소스는 편집하지 않는다. |
| encoder 준비를 Meson에 맡김 | 기존 `prepare_encoder.sh`가 재configure마다 다운로드할 수 있다. 최초 header 준비 후 build의 `encoder/` marker를 만든다. 다운로드 header는 기존 ignore 규칙을 따른다. |
| manager 새 API만 넣으면 직접 접근 | 현재 FP32 활성값 경로는 이미 외부 cache를 읽고 K RoPE/V half 변환 결과를 최종 cache에 쓴다. 이를 보존하면서 소유권과 step publication을 core로 옮긴다. native FP16의 순수 V copy는 별도 미완료 항목이다. |
| core MVP batch=1 | 기존 batch=2 FP32 manager 테스트를 유지한다. contiguous CPU storage가 batch offset을 명시적으로 검사하도록 한다. session 간 공유는 구현 범위가 아니다. |
| 모든 model 테스트가 실행됨 | 최초 baseline은 93 PASS + deprecated QINT4 fixture 1 SKIP. Meson의 suite SKIP=0 표기와 개별 gtest SKIP를 구분한다. |
| 기존 Gemma4 패키지를 바로 사용 | `skip_prefill=true`일 때 초기 layer property 오류가 발생한다. 이를 끈 별도 설정에서 FP32 `.bin`은 로드/생성이 가능하지만, 기존 양자화 `.bin`은 현재 graph와 byte 수가 다르다. 원본 파일은 수정하지 않는다. |

Host: Intel Core i9-14900, x86_64, AVX2/FMA/F16C 지원, RAM 62GiB
(착수 시 available 약 46GiB), 작업 filesystem 여유 약 69GiB.
`~/TEST_MODELS`는 읽기만 한다. 모델 검증용 설정과 symlink는 build 아래 생성한다.
외부에 설치된 nntrainer layer plugin이 기본 AppContext 초기화에서 탐색될 수 있다는
점도 확인했다. 실제 executable의 core/CausalLM 라이브러리는 `ldd`로 이 build의
경로를 확인했다. 시스템 설치를 덮어쓰지 않는다.

## 재현 가능한 빌드

저장소 루트에서 실행한다. 스크립트는 tokenizer를 현재 compiler로 먼저 build한다.
Cargo.lock을 사용하며, cached crate만 쓰려면 `KV_TOKENIZER_OFFLINE=1`을 지정한다.

```bash
git submodule update --init --recursive --depth 1 \
  Applications/CausalLM/xgrammar subprojects/googletest
KV_TOKENIZER_OFFLINE=1 CC=gcc CXX=g++ \
  tools/kv_cache/build_cpu.sh build-kv-cpu
```

실제 NNTrainer 설정 명령은 다음과 같다.

```bash
CC=gcc CXX=g++ meson setup build-kv-cpu --buildtype=debugoptimized \
  -Dplatform=none -Denable-app=true -Denable-transformer=true \
  -Denable-test=true -Denable-api=false -Denable-fp16=false \
  -Denable-opencl=false -Denable-cuda=false -Denable-npu=false \
  -Denable-htp=false -Denable-tflite-interpreter=false \
  -Denable-tflite-backbone=false -Denable-nnstreamer-backbone=false \
  -Dml-api-support=disabled -Denable-capi=disabled \
  -Denable-nnstreamer-tensor-filter=disabled \
  -Denable-nnstreamer-tensor-trainer=disabled -Dopenblas-num-threads=1 \
  -Dcausallm-tokenizer-lib="$PWD/build-kv-cpu/tokenizer/x86_64-unknown-linux-gnu/libtokenizers_c.a"
ninja -C build-kv-cpu -j8
```

재configure는 `meson configure build-kv-cpu -D옵션=값` 또는 스크립트 마지막에
`--reconfigure`를 전달한다. 스크립트의 추가 인자로 `-Denable-fp16=true
-Dfp16-impl=wrapper` 등 profile을 덮어쓸 수 있다. 서로 다른 profile은 build
디렉터리를 분리한다. runtime thread 수는 아래처럼 고정한다.

```bash
NNTR_ENGINE=cpu NNTR_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
  meson test -C build-kv-cpu --print-errorlogs \
  unittest_kv_cache_manager unittest_kv_ring unittest_causallm_models
python3 tools/kv_cache/validate_cpu_models.py \
  --build build-kv-cpu --output build-kv-cpu/validation/baseline-compatible
```

검증 스크립트는 상속된 `NNTR_*`, `LD_*` tuning을 제거하고 greedy decoding을 쓴다.
원래 설정의 `skip_prefill`은 KV prefill 자체 생략이라는 뜻이 아니라 graph의
마지막-token 최적화다. 기존 패키지의 해당 최적화 오류를 피하기 위해 smoke 기본은
false이며 `--skip-prefill`로 원래 true 조건을 재현한다.

## 변경 전 baseline

| 검증 | 결과 | 증거 |
|---|---|---|
| 전체 Linux CPU build | 로컬 tokenizer 지정 후 통과 | `/tmp/kv-baseline-build-local-tokenizer.log` |
| 기존 manager | 20 PASS | `build-kv-cpu/validation/baseline/tests.txt` |
| 기존 ring | 22 PASS | 같은 log와 XML |
| CausalLM tiny model / reference | 93 PASS, deprecated QINT4 1 SKIP | 같은 log와 XML; fixture가 전부 빠진 통과가 아님 |
| 두 Gemma4, skip_prefill=true | 각 산술/한국어 시도에서 초기화 실패: `[LayerImpl] Unknown Layer Properties count 3` | `build-kv-cpu/validation/baseline/*.stderr.txt` |
| FP32, skip_prefill=false, 산술 | `2 + 2`에 `4<turn\|>` 생성, exit 0 | `/tmp/kv-fp32-noskip.stdout`, `.stderr` |
| FP32, skip_prefill=false, 산술/한국어 재현 | 각각 `4<turn\|>`, `서울<turn\|>`; 두 실행 exit 0 | `build-kv-cpu/validation/baseline-compatible/report.json` |
| 기존 Q4_0/Q6_K, skip_prefill=false | weight 크기 불일치로 load 거절 | `/tmp/kv-model-noskip.stderr` |

양자화 파일의 실제 크기는 **2,708,362,380 bytes**, 현재 540-weight graph가 요구하는
크기는 **3,313,915,020 bytes**이며 차이는 **605,552,640 bytes**다.
이는 `262144 × (35 × 256)` PLE lookup table을 Q4_0 대신 Q6_K로 해석할 때의
크기 차이와 정확히 같다. 현재 소스는 PLE에 `embedding_dtype=Q6_K`를 적용한다.
따라서 옛 패키지의 PLE dtype 규약 차이라는 강한 가설이 있지만, 파일에 dtype
metadata가 없어 이것만으로 weight 내용을 증명한 것은 아니다. 강제로 읽거나
파일을 padding하지 않는다. 작동이 확인된 FP32 모델을 우선 사용한다.

core 구현 후 같은 산술/한국어 smoke의 비교 결과는 후속 단계에서 추가한다.
정상적인 짧은 문장만으로 HF logits 일치나 장문 정확성을 주장하지 않는다.

## 단계 경계

P0는 환경과 변경 전 baseline, P1은 내부 core storage/lease와 독립 테스트,
P2는 기존 CausalLM CPU allocation 및 graph 실행 연결이다. 각 단계에서 구현 범위,
실행한 검사, 남은 한계를 추가한다. GPU/NPU/CUDA direct access, quantized KV,
session 간 공유, Android 실기기 완료를 Linux CPU 결과로 대신하지 않는다.
