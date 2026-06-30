# YOLOv11m ORT INT8 동등 마스터플랜 (측정 근거)

목표: **e2e ≤ 208ms** (SM-S926U/SD8Gen3, ORT INT8 동일기기 실측). 현재 **1210.9ms** = 5.8× 격차.
근거 데이터: `yolov11_device_gap_analysis.md` §4b(우리 분해)·§4c(ONNX lowering). 모든 수치 per-iteration.

## 0. 확정된 출발점 (측정)

| 구간 | 현재 ms | GMAC/s | 격차원인 | 레버 |
|---|---|---|---|---|
| conv 3×3 indirect Q4_0 | 390.1 | 87.7 | requant132+GEMM116(이미 SMMLA,~430GMAC/s)+transpose142 | W4A8+NHWC |
| conv 1×1-s1 matmul Q4_0 | 371.1 | 60.8 | int8아님+transpose | W4A8+NHWC |
| conv fp 3×3 stem(conv0) | 48.7 | 6.1 | naive `__fallback_sgemm` | 양자화 or 튜닝커널 |
| depthwise 3×3 | 45.1 | 1.7 | 스칼라 | int8 NHWC dw 커널 |
| conv fp 1×1 | 13.25 | 0.3 | naive sgemm | 양자화 |
| **conv 합** | **868.7** | **65.8** | vs ORT 365 | (a) 본진 |
| slice | 119.7 | — | 스칼라 getValue 디스패치 | memcpy |
| psa_attention | 98.4 | — | 스칼라 matmul/softmax | 튜닝 GEMM+벡터softmax |
| upsample2d | 53.7 | — | 스칼라 | memcpy/벡터 |
| pooling2d | 14.8 | — | 스칼라 | 벡터 maxpool |
| **비-conv 합** | **292.6** | — | ORT는 ~12ms | (b) 위생 |
| concat / addition | 4.9 / 1.5 | — | 이미 동급 | 건드리지 않음 |
| decode/NMS | ~49 | — | detect head 스칼라 | 벡터화(P3) |

핵심: op **개수·MAC량은 ORT와 동급**. 격차는 100% **op당 속도**(커널+데이터플로우). 알고리즘/그래프 변경 불필요.

격차 2분류: **(a) conv 5.5×** = 어려운 본진(W4A8+NHWC 상호의존; **SMMLA는 이미 사용 중**, GEMM 코어만 ~430 GMAC/s로 ORT급 → 격차는 NCHW transpose+재양자화 dataflow). **(b) 비-conv ~287ms** = 독립·쉬움·numerics보존.

---

## ★ 트랙 분담 (2026-06-29)
- **비-conv 위생 트랙 = 타인 담당** (다른 분): slice 119.7 / psa_attention 98.4 / upsample2d 53.7 / pooling2d 14.8 = ~287ms. 스칼라 디스패치 → memcpy/벡터화, numerics 불변. **이 문서 범위 밖.**
- **우리 트랙 = conv 본진**: ① 출력 transpose 142ms 제거(NHWC) + ② 매 conv 재양자화 132ms 제거(W4A8 Q8_0 persist) + conv quick-win(스템/depthwise). 아래 C0~C3.
- **★두 트랙 #1 조율 포인트 = conv↔비-conv 경계(layout/dtype).** 우리가 conv를 NHWC+int8로 바꾸면 비-conv op은 NCHW+FP16 기대 → seam마다 transpose+dequant가 끼면 절감분 상쇄. 해법: (a) 비-conv도 NHWC/int8 지원(이상적, 조율 필요) / (b) conv 섬만 NHWC int8 + 경계에서만 transpose/cast(ORT의 ~15개 흉내, 저렴, 독립진행 가능). **추천: (b)로 독립 시작 → (a)로 수렴.**

## Phase C0 — conv quick-wins (독립, 리스크 낮음)
| # | 작업 | 기법 | 현재→목표 |
|---|---|---|---|
| C0.1 | conv0 fp 스템 | naive `__fallback_sgemm` 탈출(Q4_0 경로 편입 or 튜닝커널) | 48.7→~10 |
| C0.2 | depthwise | 스칼라 개선 / 이후 int8 NHWC dw 준비 | 45.1→ |

**Exit gate**: conv ~869→~810. 검출 bit-identical(C0.1만 fp 재결합 허용오차). baseline 브랜치+디바이스 A/B.

---

## Phase C1 — W4A8 기반: 층간 int8 영속 + quantize-once (병목 ② 재양자화 제거, 최대 단일레버)
**왜**: 매 conv FP→Q8_0 재양자화(indirect만 132ms/iter)+FP 활성 왕복 제거. infra 일부 프로토타입 존재(stash, [[project_w4a8_phasea_q8_0_tensor]]).

| # | 작업 | 상세 |
|---|---|---|
| 1.1 | Q8_0 활성 텐서(per-block dynamic) | ggml식 블록당 동적 스케일=캘리브 불필요, int32 누산=FP16 동적범위 벽 회피. 4D Q8_0_Tensor |
| 1.2 | conv가 int8 활성 직접 소비 | int32 누산→출력 1회 requant(ORT QLinearConv 모델). 132ms 재양자화 소거 |
| 1.3 | SiLU int8 | LUT 또는 int8 sigmoid+mul. 활성 int8 유지(FP 왕복 없음) |
| 1.4 | elementwise int8 | add/concat/slice/pool/upsample을 int8 위에서(대부분 dtype-무관 복사) |
| 1.5 | 경계 캐스트 1회 | 입력 1회 양자화·출력 1회 역양자화 |
| 1.6 | 혼합정밀 | 민감층(detect head/attention) FP16 유지, MAC 90% 차지하는 backbone conv만 int8 |

**Exit gate**: 재양자화 소거. 여전히 NCHW+SDOT+transpose. conv ~830→**~580**, e2e ~916→**~666ms**. 검출 허용오차內(int8 활성 정확도 검증=박스/mAP, FP16 baseline 대비 델타 측정).

---

## Phase C2 — NHWC (병목 ① transpose 제거 + SMMLA 풀가동, parity-maker, 가장 invasive)
**왜**: ★**정정** — SMMLA(i8mm)는 conv GEMM 코어가 **이미 사용 중**이다(`nntr_gemm_q4_0_4x8_q8_0[_fp16]` inline asm `.inst …smmla`, LLM prefill FC와 동일 커널). 커널 GEMM만 떼면 ~430 GMAC/s로 ORT(365)보다 빠르다. 따라서 P2는 "SMMLA 커널 신규작성/SDOT 교체"가 **아니다**. 진짜 일 = **NHWC로 ① 출력 transpose 142ms/iter 소거(NHWC출력=다음층 NHWC입력) + ② activation을 네이티브 channel-last로 둬서 층마다 gather/repack 안 하게** 하여, 이미 빠른 SMMLA를 dataflow 오버헤드 없이 풀로 먹이는 것. NHWC는 int8과 결합된 레버(§4: ORT도 FP32는 NCHW). ORT 기제를 그대로 역설계.

| # | 작업 | 상세 |
|---|---|---|
| 2.1 | NHWC 레이아웃(backbone) | conv 위주 backbone channels-last, weight NHWC int8 repack |
| 2.2 | NHWC용 SMMLA GEMM 경로 정비 | 신규 SMMLA 커널 작성 아님 — 기존 SMMLA indirect GEMM이 NHWC activation을 gather/repack 없이 바로 소비하도록 입력 어드레싱 조정. (GEMV decode 경로 SDOT는 무관) |
| 2.3 | 1×1 = 순수 NHWC int8 GEMM | im2col 불필요·최고 산술강도→피크 근접(현재 371ms 2위→최속군) |
| 2.4 | NHWC int8 depthwise 커널 | |
| 2.5 | 비-conv NHWC 네이티브 or 경계 transpose | pool/upsample/concat/slice NHWC, 아니면 detect head에만 경계 transpose(ORT의 15개 모방=저렴) |
| 2.6 | 출력 transpose 제거 | NCHW 산물 소거 |

**Exit gate**: conv→**~157ms**(ORT 동급), e2e ~666→**~247ms**. 검출 검증+ORT 직접 A/B.

---

## Phase C3 — 동급 초과: fusion + 튜닝 (conv epilogue)
| # | 작업 | 효과 |
|---|---|---|
| 3.1 | conv+requant+SiLU 단일 int8 epilogue 융합 | ORT는 QSigmoid+QMul 28.6ms 별도; 우린 융합→더쌈 |
| 3.2 | SMMLA 타일/스레드 튜닝(SD8Gen3) | cache blocking, K-panel packing, 스레드 밸런스. conv 157→~130 |
| 3.3 | decode/NMS 벡터화 | detect head decode+DFL 벡터화. 49→~15 |
| 3.4 | slice→conv 입력 gather 융합 | slice가 conv 먹일 때 복사 제거 |

**Exit gate**: e2e **≤208ms 초과 달성(~175ms 추정)**.

---

## 종합 projection

**우리 트랙(conv) 중심.** 비-conv(~287→타 트랙 목표 ~12)·decode는 타 트랙/별도. e2e parity는 두 트랙 합산.

| 단계 | **conv(우리)** | vs ORT conv 157 | 비-conv(타) | decode | e2e(합산) |
|---|---|---|---|---|---|
| baseline | 868.7 | 5.5× | 292.6 | 49 | **1210.9** |
| +C0 스템/dw | ~810 | 5.2× | (타 트랙) | 49 | — |
| +C1 W4A8(②재양자화 제거) | ~580 | 3.7× | (타 트랙) | 49 | — |
| +C2 NHWC(①transpose 제거, SMMLA 풀가동) | **~157** | **1.0× parity** | ~12(타 트랙 목표) | 49 | **~247** |
| +C3 fuse+tune | ~130 | 0.83× | ~12 | ~15 | **~175** |

정직한 도착점: **C1까지 conv ~580(전체의 ② 제거), parity(conv 157)엔 C2(NHWC) 필수, 초과엔 C3.** 전체 e2e 208 이하는 우리 C2/C3 + 타 트랙 비-conv 위생이 모두 안착해야 성립.

## 리스크 & 완화
- **★두 트랙 경계(layout/dtype)**: conv NHWC int8 ↔ 비-conv NCHW FP16 seam. 완화=(b) conv 섬+경계 transpose/cast 먼저(독립), (a) 비-conv NHWC/int8로 수렴. 양 트랙 인터페이스 합의 필요.
- **C1 int8 정확도**: per-block dynamic Q8_0로 캘리브 불필요+int32 누산. 그래도 detect head/attention 민감→**혼합정밀**로 backbone conv만 양자화.
- **C2 NHWC 비용(최대)**: backbone 레이아웃 영향. 완화=conv-only NHWC 섬+경계 transpose부터 점진 확대.
- **C1↔C2 결합**: NHWC가 Q8_0 block(32-elem)을 연속 채널과 정렬 → 사실상 C1의 전제. conv 전담이 된 지금 **C1+C2 통합 진행**이 재작업 적을 수 있음(단 한 번에 invasive). C0는 독립 선행.
- **SMMLA는 신규작성 불필요**: conv GEMM은 이미 SMMLA(LLM prefill과 동일 커널). 핵심은 NHWC dataflow(transpose/재양자화/gather 제거)이지 GEMM 명령 교체가 아님.

## 하지 말 것 (낭비 회피)
- 비-conv 위생(slice/attention/upsample/pool) = **타 트랙, 우리가 손대지 않음.** · FP32 sgemm 튜닝(conv0=C0.1만) · 추가 indirect/fusion 메모리 작업(완료, 속도레버 아님) · FP16-only NHWC(ORT도 FP32는 NCHW=레버 아님) · SMMLA 커널 신규작성(이미 있음).

## 검증 규율 (단계마다)
baseline 브랜치 push → 디바이스 동일기기 A/B → 검출 bit-identical(P0) 또는 허용오차+mAP(P1·P2) → RSS 기록. 측정 인스트루먼트(`YOLO_LAYER_PROFILE`/`YOLO_CONV_GEOM`/`YOLO_KERNEL_PROFILE`)로 단계별 재분해.
