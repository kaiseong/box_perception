# Box Picking 고속화 로드맵 (picking_box_5_debug.py)

> 2026-07-08 기준. 이 문서는 다음 세션(codex)이 이어서 진행할 수 있도록
> 현재 상태, 하드웨어로 검증된 사실, 남은 작업을 정리한 것이다.
> 대상 로봇: Rainbow Robotics RB-Y1 (M 모델), Jetson Orin AGX + D405.

## 0. 목표

mobile align → pre-push → push → lift 전 과정을 정지 없이(스테이지 전환 공백
0.5초 이내, 가능하면 0) 매끄럽게 잇는다. LED 기준: **초록(Executing)이 서보
시작부터 lift까지 유지**되는 것이 이상적. `picking_box_5_debug.py`가 실험장이고,
검증된 것만 `picking_box_5.py`(production)에 이식한다.

## 1. 현재 상태 (무엇이 어디까지 되나)

최근 실기 로그 기준 (2026-07-08 오전, commit `b62db68`):

| 스테이지 | 소요 | 상태 |
|---|---|---|
| 0/7 gripper_home_open | 4.2~6.1s | 병목 (P2에서 다룸) |
| 1/7 ready | 2.27s | 고정 (MINIMUM_TIME=2.0) |
| 2/7 live_vision | 1.6~1.8s | 카메라 오픈 포함 |
| 3-4/7 se2_align (서보) | **2.6~3.7s** | 필터 수정 후 안정적 settle; stationary confirm 생략 |
| → 5/7 전환 | **공백 0** | E-패턴 선점 성공 (실기 확인) |
| 5/7 vision_pre_push | **0.9~1.2s** | 새 스트림 첫 명령 + FK 게이트 (FinishCode 대기 제거) |
| → 6/7 전환 | ~0.2s (cancel) | body 스트림이 body 스트림을 선점 못해 cancel 필요 |
| 6/7 inward_push | 0.71s + FK 물림 검증 | 2026-07-08 실기에서 성공 (gap shrink 6.4~7.4cm) |
| → 7/7 전환 | ~0.1s (build→cancel→send) | 실기에서 send까지 성공 |
| 7/7 lift | 0.53s 이동 + 100s hold | FK lift gate로 성공 확인 (raised 6.7cm), FinishCode 100s 대기 제거 |

P0 결론: debug 체인은 `mobile align -> pre-push -> push -> lift`까지 한 번
완주했다. 최신 실기 총 소요는 gripper 포함 13.33s, gripper 제외 9.09s 수준이다.
다음 판단 포인트는 LED 관찰이다. pre-push→push, push→lift의 파란 blip이
0.5초 이하이면 P1/P1.5보다 P2(앞단 절대시간 단축)의 ROI가 더 크다.

## 2. 하드웨어로 검증된 rby1-sdk 규칙 (절대 다시 실험으로 배우지 말 것)

`stream_transition_probe.py`와 실기 픽 런으로 확인된 사실. 모두 FK로
실행 여부까지 검증했다 (**send 수락 ≠ 실행**임을 두 번 학습했음).

1. **스트림은 첫 명령의 컨트롤러 구성에 묶인다.** mobility로 시작한 스트림에
   body 컴포넌트를 담은 복합 명령을 보내면 수락되지만 body는 실행되지 않고,
   그 명령의 hold가 끝나면 스트림도 죽는다. (probe B)
2. **hold가 만료되면 스트림은 죽는다**: `RuntimeError: This command stream is
   expired`. 스트림 위 모든 명령은 다음 send까지의 간격보다 길게 hold해야
   한다. (probe D)
3. **일반 `robot.send_command`는 hold 중인 스트림 명령을 선점하지 못한다.**
   수락은 되지만 조용히 버려진다(discard). (probe G=False)
4. **새 스트림의 첫 body 명령은 hold 중인 일반(비스트림) 명령을 선점한다.**
   (probe E=True) → pre-push가 FK 도착 후 hold를 유지한 채 다음 스테이지가
   새 스트림으로 치고 들어오는 패턴의 근거.
5. **복합(body+mobility) 명령은 스트림의 첫 명령이면 실행된다.** (probe F=True)
6. **body 스트림은 hold 중인 mobility 스트림을 대체할 수 있다** (실기: pre-push가
   서보 브릿지를 공백 0으로 선점). 컴포넌트가 달라서 가능한 것으로 추정.
7. **body 스트림은 hold 중인 다른 body 스트림을 선점하지 못한다.** 새 스트림의
   첫 send가 `This command stream is expired`로 즉시 실패한다. (실기: streamed
   pre-push가 hold 중일 때 push 스트림 첫 send 실패)
8. Jetson에서 비전 프레임 처리(GIL ~0.3s/frame)가 30Hz 송신 스레드를 굶긴다.
   서보 스트림 명령 hold는 1.0s 이상 필요 (`SERVO_COMMAND_HOLD_TIME_SEC=1.0`,
   production에도 반영됨).

## 3. 현재 아키텍처 (debug 체인)

```
서보 (mobility 스트림, 30Hz MobileBaseServoCommandStreamer)
  └ settle → 송신 스레드 정지 → zero-vel 브릿지 (hold 0.8s) → _handoff_stream 반환
      (stationary confirm 생략; settled synced frame이 접촉 측정값)
pre-push (새 스트림의 첫 Cartesian body 명령 = 브릿지 선점, 공백 0)
  └ FK 게이트 (2cm/8°, wait_streamed_eef_arrival) → 도착
  └ cancel_control (+0.1s sleep)  ← 규칙 7 때문
push (새 스트림, 유휴 상태에서 첫 impedance 명령 = 검증된 케이스)
  └ 0.5s 램프 → 최종 타깃 hold 3s
  └ FK 물림 검증: 양손 간격 2cm 이상 축소 (PUSH_ENGAGE_MIN_GAP_SHRINK_M)
  └ 실패/expired 예외 → cancel → 유휴에서 1회 재시도 (PUSH_ENGAGE_ATTEMPTS=2)
lift (build → cancel_control → 즉시 robot.send_command, 공백 ~0.1s)
  └ production과 동일한 CartesianImpedance + 100s hold
  └ debug 완료 판단은 FinishCode 대기 대신 FK 상승 검증
     (LIFT_HEIGHT의 50% 이상 상승하면 done)
```

모든 스트림 단계에 폴백 존재: pre-push FK 실패 → cancel + 기존 send_stage 경로.
어떤 경우에도 픽은 완주하거나 안전하게 중단된다.

핵심 안전/디버깅 원칙 (이 프로젝트에서 피로 배운 것):
- **print_stage는 모든 이벤트를 stderr에 남긴다** (stdout은 [timing] 요약만).
  키워드 필터로 걸러내지 말 것 — 서보 실패 원인이 두 번 침묵당했다.
- 스트림 send 성공을 실행으로 간주하지 말 것. **반드시 FK로 검증.**
- ServoMeasurementFilter는 연속 3회 거부 시 윈도우 리셋
  (`SERVO_FILTER_REJECTION_RESET_FRAMES`). 베이스가 움직이는 동안 accepted-only
  median이 낡아 래치업되는 버그의 수정 — production에도 반영됨.

## 4. 남은 작업 (우선순위 순)

### P0 — 현재 체인 실기 검증 (**완료**)

2026-07-08 commit `b62db68` 실기에서 성공:

```
servo settled: error x=+0.9cm y=+0.0cm yaw=+1.05deg
vision_pre_push: EEF arrival confirmed by FK (pos=1.5cm rot=0.6deg)
inward_push: engaged: hand gap shrank 7.4cm (FK)
lift: engaged: raised 6.7cm by FK; command holds up to 100s
```

보존할 회귀 체크:

1. 작업 트리 커밋 → Jetson pull.
2. `python picking_box_5_debug.py --address 192.168.30.1:50051 --model m`
3. 기대 로그 순서:
   - `servo settled` → `servo bridge hold sent; stream handoff ready`
   - `sent as new-stream first command (preempts servo bridge); FK gate`
   - `EEF arrival confirmed by FK` → `arrived; cancel_control so the push stream starts from idle`
   - `engaged: hand gap shrank ~10.0cm (FK); done`
   - `cancel_control to release push hold; sending lift`
   - `waiting for FK lift engage ... not waiting for 100s hold FinishCode`
   - `engaged: raised ... by FK; command holds up to 100s`
4. LED 관찰: 초록 유지 구간과 파란 blip 위치/길이 (pre-push→push, push→lift 두 곳,
   각 0.5초 이내여야 함).
5. `FAILED`/`fallback` 메시지가 있으면 그 단계의 가정이 깨진 것 — 로그 확보.

### P1 — push→lift 공백 0 만들기 (LED blip이 문제일 때만)
아이디어: **push와 lift를 같은 impedance 스트림에서 연속 램프**로 통합.
push 스트림은 이미 30Hz로 `build_impedance_push_command(inward=...)`를
갱신한다 — 같은 컨트롤러 타입이므로 램프를 이어서
`build_dual_arm_impedance_command(inward=PUSH_DISTANCE, lift=z(t))`로 z를
올리면 컨트롤러 전환 자체가 없다. 장점: 들어올리는 동안 inward preload가
목표에 항상 남아 있어 미끄럼에도 유리.
확인/구현 사항:
- [ ] `lift` 인자는 torso 프레임 z. 토르소가 20° 피치이므로 base +z 12cm를
      만들려면 torso 프레임 delta로 변환 필요
      (`delta_torso = R(torso←base) @ [0,0,LIFT_HEIGHT]`;
      `offset_translation`은 y/z만 받으므로 빌더 확장 필요).
- [ ] 현 production lift(CartesianImpedanceControl)의 joint stiffness/damping/
      torque limit 기능이 ImpedanceControl 램프에는 없음 — 들었을 때 처짐/진동
      비교 필요. 문제되면 P1 보류하고 cancel 방식(~0.1s) 유지.
- [ ] 램프 완료 후 최종 타깃 hold를 길게 (수십 초) + 프로세스 유지
      (스트림 명령은 프로세스 종료 시 죽음 → 파지 해제됨. Ctrl+C까지 sleep).

### P1.5 — pre-push→push 공백 0 (LED blip이 문제일 때만)
규칙 7이 벽. 우회 후보 (모두 프로브 먼저):
- [ ] probe 신규 phase: **body 스트림이 hold 중인 composite(body+mobility)
      스트림을 선점하는가?** (F로 servo+pre-push를 composite 스트림으로 만들면
      마지막 hold가 composite — 이걸 impedance 스트림이 선점할 수 있으면 전체
      무공백 체인 완성)
- [ ] rby1-sdk에 스트림 명시적 종료 API가 있는지 확인 (`RobotCommandStreamHandler`
      메서드 목록). cancel_control 없이 스트림만 닫아 hold를 즉시 끝낼 수 있으면
      공백이 수 ms로 줄어든다.
- [ ] pre-push 스트림 hold를 FK 도착 예상시각에 맞춰 짧게 (예: approach+0.6s)
      설정하고, push 첫 send를 expired-재시도 루프로 hold 만료 직후 잡아채기
      (공백 = 만료까지의 잔여, 튜닝으로 0.1~0.2s 가능).

### P2 — 앞단 시간 단축 (다음 고ROI 후보)
- [ ] **카메라 파이프라인 공유**: 2/7 live_vision과 서보의 ContinuousLiveBoxView가
      각각 카메라를 열고 닫음 (오픈 ~1초씩 두 번). 한 번 열어 재사용하면 ~1.5s
      절약. 2/7을 아예 서보 안으로 흡수하는 것도 방법 (서보가 어차피 첫 프레임
      부터 측정함).
- [ ] **gripper_home_open (4.4~6.1s)**: 매 실행 호밍이 필요한가? min/max가 매번
      거의 동일 ([3.29, 2.66] / [7.83, 7.22]) — 캐시하고 스킵 옵션
      (`--skip-gripper-homing`) 또는 호밍을 ready 이동과 병렬화.
- [ ] **ready 2.27s**: MINIMUM_TIME=2.0 고정. 시작 자세가 이미 ready 근처면
      관절 오차 기반으로 단축 가능.
- [ ] 서보 파라미터: 현재 settle 1.3~3.7s로 이미 양호. kp를 더 올리기 전에
      먼저 위 항목들.

### P3 — production 이식/정리
- [ ] 실기 검증 완료된 것부터 `picking_box_5.py`로 이식 (이미 반영: filter
      래치업 수정, servo hold 1.0s). 이식 후보: E-패턴 pre-push(+FK 게이트),
      push FK 물림 검증, settled-frame 사용(=stationary confirm 생략은 debug
      전용으로 남길지 결정 — production은 confirm 유지가 보수적).
- [ ] debug에 남은 미사용 코드 정리: `build_streamed_vision_pre_push_command`,
      `MobileBaseServoCommandStreamer.stop_thread`의 zero_mobility 관련 파라미터
      등 (P1.5 결과에 따라 부활 가능성 있어 보류 중).
- [ ] 기술 부채: picking_box_3/4/5가 ~90% 중복 — picking_lib 추출.
- [ ] `tests/` 139개 통과 유지. 구조 테스트(소스 텍스트 검사)가 많으므로
      리팩터링 시 함께 갱신.

## 5. 실험 도구

`stream_transition_probe.py` — SDK 전이 규칙 검증 전용 (픽 코드에서 디버깅하지
말 것). `--phases G,E,F` (기본) / `A,B,C,D` (회귀). 새 가설은 반드시 여기에
phase로 추가하고 **FK 검증 + 원상복구 + cancel 정리**를 포함할 것. 마이크로무브
phase는 양손이 3cm 움직인다.

## 6. 실행/검증 명령 모음

```bash
# 단위 테스트 (Jetson 아닌 개발 머신에서; ROS pytest 플러그인 충돌 주의)
python -m unittest discover -s tests

# 픽 (debug 체인)
python picking_box_5_debug.py --address 192.168.30.1:50051 --model m

# 픽 (production 기준선)
python picking_box_5.py --address 192.168.30.1:50051 --model m

# SDK 전이 프로브 (팔 3cm 움직임)
python stream_transition_probe.py --address 192.168.30.1:50051 --model m
```

LED 의미 (실기 확인): **초록 = Executing(제어 활성), 파랑 = Idle(제어 없음)**.
