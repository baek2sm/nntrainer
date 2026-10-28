# Android CPU 빌드와 KV 단위 검증

2026-10-28, 사용자 지정 기기 **bacd420f**에서 Linux CPU 작업을 이어 검증했다.
모델 실행 결과와 직접 접근 범위는 [Android Qwen 검증](android-cpu-validation.md)에
별도로 기록한다. `AGENTS.md`, 원본 모델과 vendored source는 수정하지 않는다.

## 실제 환경과 선택

| 항목 | 확인값 |
|---|---|
| 기기 | `23116PN5BC`, SM8650, `arm64-v8a` |
| OS | Android 15, API 35 |
| NDK | `/opt/android-ndk-r28c`, `28.2.13676358` |
| 빌드 API / STL | android-29 / c++_shared |
| CPU ISA | `armv8.2-a+fp16+dotprod+i8mm` |
| 런타임 | `NNTR_ENGINE=cpu`, NNTrainer threads 4, OpenBLAS threads 1 |
| 전용 기기 경로 | `/data/local/tmp/nntr_kv_cpu_bacd420f` |

기기의 `/proc/cpuinfo`에서 FP16, dot product, i8mm 기능을 확인했다. 계획서의
NDK r26d를 이 환경에서 실행한 것처럼 기록하지 않고 실제 설치된 r28c를 사용한다.
엔진은 FP16을 지원하지만 모델의 activation dtype은 FP32다. 기존 Android attention은
이 입력을 내부에서 FP16으로 좁히므로 Linux와의 bitwise 일치를 전제로 하지 않는다.

## 재현 빌드

초기화한 dependency는 기존 xgrammar/googletest와 고정 revision의 iniparser다.
OpenBLAS는 저장소가 지정한 `openblas-0.2.20.tar.gz`를 builddir에 준비했다.
tokenizer는 in-tree Cargo.lock과 설치된 Android Rust target으로 별도 빌드했다.

```bash
git submodule update --init --recursive --depth 1 \
  Applications/CausalLM/xgrammar subprojects/googletest subprojects/iniparser
ANDROID_NDK=/opt/android-ndk-r28c \
  Applications/CausalLM/tools/build_tokenizers_c.sh --target=android --offline \
  --out-dir="$PWD/build-kv-android/tokenizer"
cp build-kv-android/tokenizer/aarch64-linux-android/libtokenizers_c.a \
  Applications/CausalLM/lib/libtokenizers_android_c.a

ANDROID_NDK=/opt/android-ndk-r28c ANDROID_SERIAL=bacd420f \
  Applications/CausalLM/build_android.sh --nntr-threads=4 \
  -Denable-app=false -Denable-test=false \
  -Denable-opencl=false -Denable-cuda=false -Denable-htp=false \
  -Denable-tflite-interpreter=false -Denable-tflite-backbone=false \
  -Denable-capi=disabled -Dml-api-support=disabled \
  -Denable-nnstreamer-tensor-filter=disabled \
  -Denable-nnstreamer-tensor-trainer=disabled
```

`enable-app/test=false`는 엔진 패키징 단계의 옵션이다. CausalLM 앱은 이어지는
canonical `app_build/meson.build` graph로 별도 cross compile된다. 생성물은
`builddir/android_build_result/`와 `builddir_app/cpu/`에 있다. 앱 소스 수정 후에는
`ninja -C builddir_app/cpu -j8`로 갱신할 수 있다.

기존 prepare_encoder의 재다운로드를 피하기 위해 이미 header가 준비된 builddir의
`encoder/` marker를 유지했다. tokenizer와 모델·runtime·로그는 ignore되는 생성물이며
추적 중인 x86 tokenizer archive를 교체하지 않았다.

## ABI와 probe

생성된 `nntrainer-abi.ini`는 다음 flags를 앱과 테스트에 전달한다.

```text
-march=armv8.2-a+fp16+dotprod+i8mm -DENABLE_FP16=1 -DUSE__FP16=1
```

root와 standalone Meson graph의 main object 이름이 달라 기존 probe builder를
일반화했다. 실제 `compile_commands.json`의 `-o` object로 target과 link 치환을
결정하며 Android에는 host 절대경로 RUNPATH를 추가하지 않는다.
관찰용 probe TU는 `-fno-fast-math`로 컴파일해 `std::isfinite` 검사가 상수 true로
최적화되지 않게 한다. 엔진/모델의 연산 flags를 바꾼 것은 아니다.

```bash
python3 tools/kv_cache/build_qwen_cpu_probe.py --build builddir_app/cpu \
  --output build-kv-android/validation/baseline/qwen_cpu_probe
```

Linux root graph와 Android standalone graph 양쪽에서 probe compile/link를 확인했다.
baseline의 네 library, probe, ABI, source/hash metadata는
`build-kv-android/validation/baseline/`에 보존했다. 다른 ABI의 header와 library를
섞지 않는다.

## 기기 KV 단위 검증

JNI harness에 core/facade/ring executable을 추가하고 private `<kv_cache/...>`
include를 연결했다. 각 테스트가 자체 main을 가지므로 gtest-all과 gtest_main을
분리했다. snapshot fixture는 Android에 없는 `/tmp` 대신 `TMPDIR` 또는 cwd를 쓴다.

```bash
ANDROID_NDK=/opt/android-ndk-r28c /opt/android-ndk-r28c/ndk-build \
  -C Applications/CausalLM/jni NDK_PROJECT_PATH=. NDK_LIBS_OUT=./libs \
  NDK_OUT=./obj APP_BUILD_SCRIPT=./Android.mk \
  NDK_APPLICATION_MK=./Application.mk \
  CAUSALLM_PREBUILT_LIB=../../../builddir_app/cpu/libcausallm.so \
  unittest_kv_cache_core unittest_kv_cache_manager unittest_kv_ring -j4
```

NDK의 googletest source를 기존 harness의 ignored `jni/googletest/` 위치에 준비했다.
모든 ADB 호출은 `adb -s bacd420f`를 사용하며, 배포와 실행은 전용 `checks/` 경로에서
진행했다. `TMPDIR`와 `LD_LIBRARY_PATH`도 해당 절대경로로 고정했다.

| 기기 실행 | 결과 |
|---|---|
| core geometry/storage/lease | 18 PASS |
| CausalLM facade/snapshot | 25 PASS |
| ring geometry | 22 PASS |
| 합계 | **65 PASS, 0 FAIL, 0 SKIP** |

변경된 fixture의 Linux facade 25개도 통과했다. Android XML, stdout, executable hash와
ABI 기록은 `build-kv-android/validation/checks/validation.json`과 같은 디렉터리에 있다.
이 단위 검증 통과만으로 attention 직접 접근 완료를 판단하지 않는다. Qwen 실제 모델의
주소·copy 계측과 출력 검증은 별도 단계다.
