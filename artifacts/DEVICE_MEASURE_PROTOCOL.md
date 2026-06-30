# 디바이스 측정 조율 프로토콜 (3-에이전트 공유)

> **상황**: 세 워크스페이스(`nntrainer`, `nntrainer_ref`, `nntrainer_ref2`)가 동일 task를
> 각자 진행한다. **물리 ADB 디바이스는 1대뿐**이라 동시 추론은 OOM/측정오염을 부른다.
> 그래서 **디바이스 측정만큼은 한 번에 하나**로 직렬화한다. (빌드·코딩·x86 유닛테스트는 자유.)

## 공유 위치 (세 워크스페이스 바깥)
```
/home/seungbaek/projects/device_lock/
  dev_acquire.sh        # 락 획득 (원자적 mkdir 뮤텍스, 스테일 30분 자동회수)
  dev_release.sh        # 락 해제 + 결과 한 줄 로그
  device_measure_log.md # append-only trace (누가/언제/어떤 task/결과)
  .lock/                # 점유 중일 때만 존재 (holder.txt = 현재 점유자)
```

## 절차 (모든 디바이스 측정 전후 필수)

### 1) 측정 직전 — 락 시도
```bash
/home/seungbaek/projects/device_lock/dev_acquire.sh <agent_id> <task_id> <serial>
#   <agent_id> 예: ws-ref2 / agentB (워크스페이스나 본인 식별자)
#   <task_id>  예: A1, C3 ...
#   <serial>   adb devices 의 기기 serial
```
- 출력 **`ACQUIRED`** (exit 0) → 곧장 측정 진행. 측정은 **짧게**(8 iter) 끝내고 즉시 해제.
- 출력 **`BUSY by: ...`** (exit 1) → **남이 측정 중.** 아래 2)로.

### 2) BUSY일 때 — 다른 작업 먼저, 그러나 스킵 금지
- 디바이스가 필요 없는 일로 **전환**해서 진행한다:
  - 코드 구현 / 리팩터 / 빌드(`ninja -C build_android`) / **x86 유닛테스트(G-UNIT)** / 리포트 초안 작성 등.
- **일정 간격(약 5분)으로 `dev_acquire.sh` 재시도.** 비면 즉시 잡아서 측정.
- ❗**측정을 영구히 건너뛰지 말 것.** work order는 A/B 디바이스 수치(검출·ms·RSS)가
  채워져야 "완료". 지금 못 재면 *나중에 반드시* 재서 채운다.

### 3) 측정 직후 — 즉시 해제
```bash
/home/seungbaek/projects/device_lock/dev_release.sh <agent_id> <task_id> "<한줄결과>"
#   예: "... A1 base 1210.9ms→B 1180.2ms, det 0.9258/0.8868 G-BIT PASS, RSS 226MB"
```
- 해제를 잊으면 다른 둘이 영영 못 잰다. 측정 끝나면 **반드시** release.
- 비정상 종료로 락이 남아도 30분 지나면 다음 acquire가 스테일로 자동 회수(로그에 STALE-RECLAIM 남음).

## 규칙 요약
- **한 번에 한 측정.** acquire 없이 `yolov11_infer`/`adb shell ... infer` 실행 금지.
- **점유 시간 최소화.** 빌드·분석은 락 **밖**에서. 락 안에서는 추론 실행만.
- **모든 측정은 로그에 남는다.** `device_measure_log.md`로 누가 언제 무엇을 쟀는지 추적 가능.
- 측정 결과 자체(상세 수치)는 각 work order 리포트(§4 양식)에, 로그엔 한 줄 요약만.
