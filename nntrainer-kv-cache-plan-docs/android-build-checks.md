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

| 초기 baseline runtime 기기 실행 | 결과 |
|---|---|
| core geometry/storage/lease | 18 PASS |
| CausalLM facade/snapshot | 25 PASS |
| ring geometry | 22 PASS |
| 초기 합계 | **65 PASS, 0 FAIL, 0 SKIP** |

변경된 fixture의 Linux facade 25개도 통과했다. Android XML, stdout, executable hash와
ABI 기록은 `build-kv-android/validation/checks/validation.json`과 같은 디렉터리에 있다.
이 단위 검증 통과만으로 attention 직접 접근 완료를 판단하지 않는다. Qwen 실제 모델의
주소·copy 계측과 출력 검증은 별도 단계다.

## 실제 모델 초기화에서 발견한 통합 문제

batch=1 / inference guard를 추가한 첫 실제 Qwen FP32·Q4 smoke 2개가 추론 전에
초기화에서 거절되었다. 조사 과정에서 `LayerNode::refinalize()`가
`InitLayerContext`를 만들 때 graph의 execution mode를 전달하지 않아 기본 `TRAIN`을
사용하는 기존 오류를 확인했다. `compile(INFERENCE)`를 호출해도 재초기화 context는
달라질 수 있었다. graph mode를 refinalize까지 전달하도록 엔진 코드를 수정했으며,
앱만 교체해서 반영되는 변경이 아니므로 Android 엔진과 앱을 함께 다시 빌드했다.

이미 구성된 동일 build 디렉터리에서는 다음 명령을 실제로 사용했다.

```bash
ninja -C builddir -j8
meson install -C builddir --no-rebuild --skip-subprojects
ninja -C builddir_app/cpu -j8
```

이 mode 오류는 host regression으로 재현했다. 수정 전 INFERENCE·VALIDATE 두 case가
실패했고 수정 후 mode 보존 및 기존 training 기본 동작 4개가 모두 통과했다.
증거는 `checks/refinalize-mode.before.log`, `refinalize-mode.after.log/.xml`에 있다.
첫 실제 모델 초기화 거절 2개는 `final-guarded/`의 per-model stderr·report에 보존했다.
첫 로그에는 실제 context mode를 출력하지 않았으므로 두 거절 모두의 직접 원인이
mode였다고 단정하지 않는다.

mode를 보존한 다음 진단 runtime에서는 실제 Qwen 8개 case가 여전히 초기화에서
거절되었다. 이번 메시지는 `process_engine=cpu`, `layer_engine=1`,
`execution_mode=1`, `batch=1`이었다. CausalLM의 residency policy는 `mha_core`를
engine-neutral type으로 선언하며 `LayerNode::finalize()`는 이 context에 GPU stamp를
부여한다. 이 값은 CPU attention의 실제 실행 backend를 뜻하지 않는다. 잘못 추가한
context-engine CPU 검사를 제거하고 실제 runtime 선택인 `NNTR_ENGINE=cpu` 조건을
유지했다. 두 번째 진단의 8개 실패는 `final-verified/`에 보존했다. 이 10개 거절은
추론 출력을 생성하지 않았으며 성공 모델 검증 횟수에 포함하지 않는다.

## 최종 final-cpu runtime 재검증

Android FP32 attention opt-in은 CPU process, batch=1, inference, 외부 FP16 cache와
FP32 Q/K/V activation을 요구한다. profile suite는 CPU context의 지원 profile,
CPU process에서 engine-neutral GPU context 허용, 명시적 `0`의 legacy 유지 3개와
training/validation, batch=2, 비CPU process, half Q/K/V, FP32 cache, 내부 cache,
RoPE 없음, INT8, sink 거절 12개로 구성한다. **15개가 실제 기기에서 모두 통과**했다.
이 target은 production `mha_core.cpp`를 직접 컴파일한다. 테스트용 context만으로
전체 graph 통합을 대신하지 않으며 실제 Qwen 검증은 별도 문서에 기록한다.

mode 전달 수정과 engine-neutral context 허용을 모두 반영한 최종 runtime을
`build-kv-android/validation/final-cpu/runtime/`에 고정했다. 해당 디렉터리의 네 library를
기기의 전용 `checks/`에 배포하고 네 test target을 다시 빌드·실행했다. 로컬 frozen
파일과 기기 library·실행 파일의 SHA256이 모두 일치한다.
`final-cpu/build.json`은 반영된 source 파일 및 runtime hash를 함께 보존한다.

```text
libnntrainer.so: 4,960,352 bytes
SHA256: 0b7204882344ffe917cd1fd4e991e84e7e0930f0ad5611fbf31a1727d7479ad8
libcausallm.so: 17,591,568 bytes
SHA256: 78502cc34a2ee950e3d177186a0bb3c43950376e4df3f6daa1aea21ca24f0cd7
```

| 최종 final-cpu 기기 단위 검증 | 결과 |
|---|---|
| core geometry/storage/lease | 18 PASS |
| CausalLM facade/snapshot | 25 PASS |
| ring geometry | 22 PASS |
| Android FP32 profile 승인·거부 | 15 PASS |
| 합계 | **80 PASS, 0 FAIL, 0 SKIP** |

네 실행 파일 모두 exit code `0`이며 disabled·error도 `0`이다. 증거는
`build-kv-android/validation/checks/final-cpu.validation.json`,
`final-cpu.device.sha256.txt`, `final-cpu.ndk-build.log`와
`final-cpu.unittest_*.xml/.log`에 있다. manifest는 네 suite의 실제 결과와 exit code,
source/executable/runtime/device hash, XML·log hash 및 ADB 명령을 기록한다.
기존 profile 15개 결과와 다른 runtime의 65개 결과를 합산한 수치가 아니라 네 suite를
같은 최종 runtime으로 다시 실행한 결과다.

재현용 build와 기기 실행 명령은 다음과 같다. 여기서 단위 테스트는 모델을 로드하지
않고 전용 `checks/`만 사용한다.

```bash
ANDROID_NDK=/opt/android-ndk-r28c /opt/android-ndk-r28c/ndk-build \
  -C Applications/CausalLM/jni NDK_PROJECT_PATH=. NDK_LIBS_OUT=./libs \
  NDK_OUT=./obj APP_BUILD_SCRIPT=./Android.mk \
  NDK_APPLICATION_MK=./Application.mk \
  CAUSALLM_PREBUILT_LIB=../../../builddir_app/cpu/libcausallm.so \
  unittest_kv_cache_core unittest_kv_cache_manager unittest_kv_ring \
  unittest_mha_cpu_profile -j4

adb -s bacd420f push \
  build-kv-android/validation/final-cpu/runtime/libnntrainer.so \
  build-kv-android/validation/final-cpu/runtime/libccapi-nntrainer.so \
  build-kv-android/validation/final-cpu/runtime/libcausallm.so \
  build-kv-android/validation/final-cpu/runtime/libc++_shared.so \
  /data/local/tmp/nntr_kv_cpu_bacd420f/checks/

for suite in unittest_kv_cache_core unittest_kv_cache_manager unittest_kv_ring \
  unittest_mha_cpu_profile; do
  adb -s bacd420f push \
    "Applications/CausalLM/jni/obj/local/arm64-v8a/$suite" \
    /data/local/tmp/nntr_kv_cpu_bacd420f/checks/
  adb -s bacd420f shell \
    "cd /data/local/tmp/nntr_kv_cpu_bacd420f/checks && chmod 755 $suite && \
    LD_LIBRARY_PATH=/data/local/tmp/nntr_kv_cpu_bacd420f/checks \
    TMPDIR=/data/local/tmp/nntr_kv_cpu_bacd420f/checks NNTR_ENGINE=cpu \
    ./$suite --gtest_color=no --gtest_output=xml:final-cpu.$suite.xml" \
    > "build-kv-android/validation/checks/final-cpu.$suite.log" 2>&1
  adb -s bacd420f pull \
    "/data/local/tmp/nntr_kv_cpu_bacd420f/checks/final-cpu.$suite.xml" \
    build-kv-android/validation/checks/
done
```

이전 baseline 65개, 최초 profile 15개, 중간 guarded 65개는 기존 파일에 유지하며,
잘못된 context guard가 있던 중간 80개는
`checks/history/final-verified-context-guarded/final-verified.validation.json`과
인접 XML·로그에 별도 보존했다. 최종 결과는 `final-cpu.*`만 가리킨다.
