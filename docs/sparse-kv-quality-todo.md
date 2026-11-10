# Sparse KV cache 분석과 실험 재현 안내

이 브랜치는 **중간이 삭제된 요청에서도 이전 KV를 재사용하는 구현**과 후속 품질 실험을
검토하기 위한 기준 구현이다. NNTrainer의 실제 캐시 경로와 Quick.KV의 HF 실험 경로를
각각 재현하고 비교할 수 있다. 선택적 재계산은 실험 후 도입 여부를 결정할 TODO다.
구현과 기기 측정 기준은 `50355fcd8`이다.

## 1. 현재 구현은 무엇을 하는가

```text
저장된 source: A B C D
새 요청:      A   C D E
exact:        A 재사용 → C D E 계산
sparse:       A C D의 KV gather → K 위치 보정 → E 계산
```

A–E는 token 또는 연속 구간이다. 같은 source history에서 token ID 순서를 유지해
매칭하고, K에는 `R(dst-src)`를 적용하며 V는 복사한다. **삭제된 문맥의 영향은 KV에
남으므로 근사 재사용**이다. exact보다 64행 이상 더 재사용할 때 HTP sparse 경로를
선택하고, Q8 calibration을 갱신한다. approximate 결과는 exact cache에 게시하지 않는다.
Sparse source는 필요한 행만 owned page로 보존한다. 기능은 기본 OFF다.

읽을 코드는 다음 순서가 편하다.

| 역할 | 코드 |
| --- | --- |
| CPU 매칭 → gather·RoPE | [kv_prefix_cache.cpp](../nntrainer/kv_cache/kv_prefix_cache.cpp) → [kv_cache_manager.cpp](../nntrainer/kv_cache/kv_cache_manager.cpp) |
| HTP exact/sparse 선택 | [DeviceKVManager::reuse](../Applications/CausalLM/device_kv_manager.h) |
| HTP 복원·calibration·source 보존 | [CausalLM::restoreSparseDevicePrefill / captureDevicePrefill](../Applications/CausalLM/models/causal_lm.cpp), [gatherSparsePrefix](../Applications/CausalLM/kv_cache_manager.cpp) |

CPU는 FP32/FP16 및 packed TurboQuant를 지원한다. 기기 검증 대상은 Gemma4 Q8이며,
HTP KV session과 TurboQuant 병용은 지원하지 않는다. [설정과 상세 계약](sparse-kv-prefix-cache.md)을 참고한다.

## 2. 현재까지 확인한 효과

S26 Ultra, Gemma4 26B A4B, NPU attention/MoE, Q8, prefill 4096행,
각 모드 2회 중앙 TTFT(초). exact/sparse는 같은 source를 사용해 순서를 교차했다.

| 요청 | 기존 exact | Sparse | exact 대비 |
| --- | ---: | ---: | ---: |
| 24K source에서 4K 삭제 → 20K 유지, 첫 삭제 지점 4K | 38.926 | 4.265 | 9.13× |
| 24K source에서 1K 구간 네 곳 삭제 → 20K 유지 | 42.226 | 4.619 | 9.14× |
| 24K source에서 토큰 하나 삭제 | 56.088 | 5.087 | 11.03× |

첫 삭제 이후의 KV도 재사용해 prefill을 생략한 효과다. 4K/12K/24K source 보존량은
합계 **1,080 MiB**이며, 과거 1024행 실험의 2,880 MiB와 구분한다. weights·active KV는 제외한다.

검증은 exact/sparse **32건**, cold **12/16건**, 호스트 테스트 **76+37개**까지 완료했다.
완료한 cold 출력은 exact와 모두 일치했고, positive sparse 14개 출력은 요청한 사실을 유지했다.
일부 sparse 출력 ID는 cold와 달랐다. cold 경로에서도 ADB가 끊겨 장시간 안정성은 미확정이다.
남은 cold 4건·원본 DSP 비교·실모델 pre-baked 검사는 미완료다.

`5e97254` 호스트 엔진/CausalLM과 현재 버전을 **같은 현재 DSP**로 비교한 20K 재계산은
59.451/59.475초였다. 이번에는 선풍기가 없어 과거 46.829초와 직접 회귀 비교하지 않는다.
[전체 결과·원시 근거·미완료 범위](sparse-kv-prefix-4096-validation.md)에 측정 조건을 보존했다.

기기 재현은 기존 Q8 모델과 일치하는 Android 빌드에서 다음 순서로 한다.

```sh
htp_prefill_check
htp_sparse_benefit MODEL_DIR cached.json 2 all cache 4096
htp_sparse_benefit MODEL_DIR cold.json 2 all cold 4096
python3 tools/kv_cache/summarize_sparse_benefit.py cold.json cached.json summary
```

집계기는 두 실행이 모두 완료돼야 통과한다. 빌드·실행 환경은 [상세 재현 절차](sparse-kv-prefix-4096-validation.md#reproduction)를 따른다.

## 3. Quick.KV에서 비교하는 품질 개선 개념

문제는 위치 보정 후에도 남는 문맥 오차다. 다음은 **연구 후보**이며 NNTrainer에 추가 구현한 기능이 아니다.

| 개념 | 비교하려는 것 | 실험 방법 / 주의점 |
| --- | --- | --- |
| Superset / independent | 전체 문맥에서 만든 KV와 블록별 독립 KV 중 어느 쪽이 유리한가 | `cacheblend_superset`, `cacheblend_independent`; 같은 선택 예산으로 비교 |
| 삭제 직후 복구 | 오차가 커지는 구간부터 계산하면 효율적인가 | `gap`은 비율 예산, `gap_chunk16/32/64`는 삭제 직후 고정 창. 선택 토큰은 mixed causal KV로 실제 재계산 |
| 초기 레이어 활용 | fresh/cached K 차이로 복구 위치를 고를 수 있는가 | CacheBlend는 check layer 1에서 선택. `early_layers1/2/4/8`은 초기 레이어만 전체 계산하는 대조군 |
| Gradient 중요도 | superset만 미리 미분한 중요도로 위치를 고를 수 있는가 | `grad_tail_scaled` 등. offline backward 비용도 별도 계산 |
| Oracle 진단 | 선택기가 나쁜지, 부분 재계산 자체의 한계인지 | `oracle_diff`는 cold KV 차이로 선택, `oracle_copy`는 GT KV 복사. 배포 정책·지연 순위로 해석하지 않음 |
| 저랭크 residual | KV 차이를 작은 basis와 계수로 보정할 수 있는가 | 별도 legacy 실험. held-out 계수도 cold KV로 구해 아직 oracle 의존; 공통 벤치에는 미통합 |

긴 JSON 응답 파일럿(8 masks × 1 query, Qwen3-0.6B CPU FP32)에서는 RoPE의 cold 전체 출력
일치 **1/8**이 chunk 64에서 **6/8**로 늘었지만, 재계산 비중도 약 **69%**였다.
이는 비용이 같은 조건의 우위나 일반화 증거가 아니다. KV 오차·출력 보존·정답률을 따로 보고,
동일한 실제 token×layer 계산량과 지연에서 비교해야 한다.

## 4. 다른 PC에서 Quick.KV 기준 실험 재현

로컬 Quick.KV의 미커밋 파일에 의존하지 않도록 검증했던
[소스 묶음](validation/quickkv-sparse-reference-20261110.tar.gz)을 첨부했다.
HF reference·spec·테스트·과거 파일럿 설정/집계가 들어 있으며, 모델 weights와 serving connector는 없다.
[SHA256·버전 기록](validation/sparse-quality-provenance-20261110.json)으로 확인할 수 있다.
과거 실행 이후 backend/runner가 바뀐 snapshot이므로 **동일 조건 재평가용**이며 과거 숫자의 정확한 replay는 아니다.

Python 3.10 환경에서 저장소 루트부터 실행한다. 모델 없이 테스트와 manifest 생성까지 가능하다.
실모델 실행 전에는 아래 revision의 Qwen3-0.6B를 로컬 HF cache에 준비한다.

```bash
REF=$(mktemp -d /tmp/quickkv-reference-XXXXXX)
tar -xzf docs/validation/quickkv-sparse-reference-20261110.tar.gz -C "$REF"
cd "$REF"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.13.0 transformers==5.14.1 tokenizers==0.22.2 \
  numpy==2.2.6 matplotlib==3.10.9 pytest==9.1.1
python -m pytest tests/benchmarks/test_sparse_bench.py -q
python -m benchmarks.sparse_bench plan \
  --spec benchmarks/sparse_bench/device_tools_json_trace.json --seed 0 --out manifest.json
HF_HUB_OFFLINE=1 python -m benchmarks.sparse_bench run --manifest manifest.json --out run \
  --model Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --device cpu --dtype float32 --threads 4 --repeats 3 --selection-seed 0 \
  --methods rope gap cacheblend_superset gap_chunk16 gap_chunk32 gap_chunk64 --budgets .15 \
  --masks m3af,m0cf,m2aa,m096,m0c4,m190,m006,m005 --check-layer 1 \
  --include-absent --query-families wording-2 --query-limit 1 \
  --max-tokens 96 --min-reference-tokens 32 --heads
python -m benchmarks.sparse_bench report --run run --out report --detail-methods all
```

`report/index.html`에서 토큰·레이어별 오차와 생성 보존을 보고, `run.json`에서 환경·hash를 확인한다.
방법 목록을 바꾸면 위 표의 다른 공통 벤치 개념도 비교할 수 있다. 상세 계약은 묶음 안의
`benchmarks/sparse_bench/README.md`에 있다. HF 조립/재계산 시간은 storage I/O나 NPU serving TTFT가 아니다.

후속 TODO: **동일 예산 비교 → train/dev에서 정책 선택 → held-out mask·query 평가 → NNTrainer 반영 여부 결정**.
`--query-limit 1`은 파일럿용이므로 본평가에서는 제거한다. 재계산 경로 도입은 그 결과를 보고 결정한다.
