# Sparse KV cache 품질 개선 TODO

기준: NNTrainer `9bf6f1fad`, 2026-11-10. 목표는 삭제 전 문맥이 남은 KV의
오차를 줄이면서 cold 대비 출력 보존과 실제 TTFT의 절충점을 찾는 것이다.

## 현재 구현과 개선 지점

`KvPrefixCache::matchSparse`는 단일 radix history에서 token ID의 순서를 보존해
최장 요청 prefix를 찾는다. `KvCacheManager::reuseSparsePrefix`는 새 페이지에
gather하고 K에 `R(dst-src)`를 적용하며 V는 복사한다. FP32/FP16 및 CPU
TurboQuant를 지원하지만, **위치 보정으로 삭제된 문맥의 attention 영향은 제거되지 않는다.**
HTP의 `DeviceKVManager::reuse` → `CausalLM::restoreSparseDevicePrefill`은
exact보다 64행 이상 이득일 때 sparse를 선택하고 Q8 calibration을 무효화한다.
현재는 중간 선택 토큰 재계산 없이 복원 뒤 suffix를 계산한다. approximate 이력은
exact cache에 게시하지 않는다. HTP KV session과 TurboQuant의 병용은 지원하지 않는다.
상세 동작·실기기 성능은 [기존 구현 문서](sparse-kv-prefix-cache.md)를 참고한다.

## 우선순위별 후속 작업

- [ ] **P0 — 재현 기준 고정.** Quick.KV `d8f8626197c4734e67d8fbac959ac572302ed1fa`의
  실험 디렉터리는 현재 untracked다. 외부 재현 전에 소스·spec·원시 결과를 별도
  commit/artifact로 공개하고 아래 provenance와 대조한다. HEAD만 checkout하면 부족하다.
  cold 반복, native exact continuation, 전 토큰 재계산의 KV/logits 수치 기준선을 먼저 통과시킨다.
- [ ] **P1 — 토큰 경계와 재계산 경로.** 전체 요청을 자연 토큰화하고 블록·문자 offset·ID로
  매핑한다. 매핑 불가능한 경계는 필수 재계산 비용으로 분리한다. NNTrainer의 개별 ID
  subsequence 매칭은 도구 블록을 구분하지 않으므로 반복 ID의 블록 오매칭도 검증한다.
  선택 토큰을 embedding부터 레이어별 mixed causal KV에 attend시키고 새 K/V를 기록하는
  경로를 추가한다. GT 복사와 구분하고 source 불변성·approximate provenance를 유지한다.
- [ ] **P2 — 삭제 직후 복구부터 비교.** `rope`, `random`, `gap@{.05,.15,.30}`와
  `gap_chunk{16,32,64}`를 비교한다. 창은 실제 삭제 run 직후부터 시작해 병합한다.
  선택 행의 원본 KV 로드를 생략하고 Q8 재보정·sliding/global attention·마지막 prompt
  token의 fresh 평가를 검증한다. 고정 창과 비율 방식은 실제 token×layer 비용을 맞춰 비교한다.
- [ ] **P3 — 선택기 확장.** `cacheblend_superset`은 check layer 1(0 기반)의 fresh/cached K
  차이로 전역 선택하며 초기 전체-prefix 계산 비용이 추가된다. `early_layers{1,2,4,8}`도
  대조하되 초기 계산만으로 뒤쪽 loaded KV가 고쳐진다고 가정하지 않는다. superset-only
  gradient와 저랭크 residual은 후순위로 둔다. GT diff/출력-target gradient는 oracle 진단용이다.
- [ ] **P4 — 품질과 실기기 승인 기준.** train/dev에서 정책을 정하고 held-out mask와
  wording-2, present/absent 쿼리로 평가한다. 전체 token ID 일치, 공통 prefix,
  16/32/64/96-token 보존, 동일 cold history의 full-vocabulary TVD/KL, exact tool 정답률을
  분리한다. mask bootstrap CI와 worst case를 보고하고, Gemma4 Q8에서 cold/exact/sparse
  교차 반복으로 TTFT·decode·KV 메모리·실제 I/O를 측정해 비용에 따른 fallback을 정한다.

## 실험에서 가져올 근거

Quick.KV 긴 JSON 파일럿: Qwen3-0.6B, CPU FP32, 8 masks × 첫 쿼리 1개,
cold 63–96 tokens, 생성 상한 96. 다음 값은 **기존 결과**이며 현재 코드 재측정값은 아니다.

| 방식 | cold 전체 출력 일치 | 평균 공통 prefix | 재사용 후보 중 선택 비율 | 평균 teacher TVD |
| --- | ---: | ---: | ---: | ---: |
| RoPE | 1/8 | 44.64% | 0% | .05415 |
| CacheBlend superset 15% | 1/8 | 49.90% | 15.30% | .04479 |
| 삭제 직후 chunk 64 | 6/8 | 85.26% | 69.30% | .01053 |

출처: Quick.KV `benchmarks/results/sparse_bench_json_trace_v1/chunk_repair_report_v1/chunk_findings.json`.
chunk 64는 계산량이 크며 짧은 응답 결과와 합산할 수 없다. 기존 0.6B 10-case/12-token
실험은 k=8/tool 대응 예산에서 superset key 선택 9/10, independent 2/10이었다
(`docs/cacheblend-comparison.md`). 독립 캐시보다 superset을 우선할 근거이지만 일반화
증거는 아니다. system-gradient/oracle 및 저랭크 보정 결과도 각각 조건·GT 의존성이 다르다.

## 로컬 재현 명령

Quick.KV 루트에서 실행한다. Python 3.10.12, torch 2.13.0+cu132,
transformers 5.14.1, numpy 2.2.6; report에는 matplotlib도 필요하다.
해당 환경과 로컬 HF 모델 snapshot을 준비하고 새 출력 경로를 사용한다.

```bash
cd ~/projects/Quick.KV
PY=~/projects/vllm/.venv/bin/python
OUT=$(mktemp -d /tmp/sparse-quality-XXXXXX)
export HF_HUB_OFFLINE=1
$PY -m benchmarks.sparse_bench plan --spec benchmarks/sparse_bench/device_tools_json_trace.json --seed 0 --out "$OUT/manifest.json"
$PY -m benchmarks.sparse_bench run --manifest "$OUT/manifest.json" --out "$OUT/run" \
  --model Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --device cpu --dtype float32 --threads 4 --repeats 3 --selection-seed 0 \
  --methods rope gap cacheblend_superset gap_chunk16 gap_chunk32 gap_chunk64 --budgets .15 \
  --masks m3af,m0cf,m2aa,m096,m0c4,m190,m006,m005 --check-layer 1 \
  --include-absent --query-families wording-2 --query-limit 1 \
  --max-tokens 96 --min-reference-tokens 32 --heads
$PY -m benchmarks.sparse_bench report --run "$OUT/run" --out "$OUT/report" --detail-methods all
```

manifest SHA256는 `7b3123ed64e53928503a966f1ef34580fb4117bfd925c44a45397a67c9dd59e0`.
현재 소스로 같은 조건을 다시 평가하는 명령이다. backend/runner가 기존 실행 이후
변경되어 과거 수치의 정확한 replay에는 당시 소스가 필요하다. `--query-limit 1`은
파일럿 재현용이며 P4 본평가에서는 제거한다. 측정 시간은 resident tensor 조립·선택·복구·
재계산이고 storage I/O나 serving TTFT가 아니다. 모델·tokenizer·소스 해시와 집계값은
[재현 기록](validation/sparse-quality-provenance-20261110.json)에 보존한다.
