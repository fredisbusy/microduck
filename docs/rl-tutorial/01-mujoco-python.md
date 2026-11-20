# 1단계: Python으로 시뮬레이션 돌리기

[← 0단계](00-mujoco-xml.md) · [목차](README.md) · [다음: 2단계 →](02-rsl-rl-ppo.md)

## 목표

- `mujoco` Python API로 시뮬레이션을 한 스텝씩 돌린다.
- 강화학습 환경의 세 부품인 **관측·보상·종료 조건**을 함수로 직접 짠다.
- 무작위 행동으로 몇 판 돌려서 "학습 전 실력"을 잰다.

## 알아 둘 개념

### 모델과 데이터

| 객체 | 뜻 | 바뀌는가 |
|---|---|---|
| `MjModel` | XML을 읽은 결과. 질량, 관절 범위, 모터 세기 같은 **설계도** | 안 바뀜 |
| `MjData` | 지금 이 순간의 **상태**. 위치, 속도, 모터 입력 | 매 스텝 바뀜 |

자주 쓰는 `MjData` 필드:

| 필드 | 뜻 | CartPole에서 |
|---|---|---|
| `data.qpos` | 관절 위치 | `[수레 위치 x (m), 막대 각도 θ (rad)]` |
| `data.qvel` | 관절 속도 | `[수레 속도, 막대 각속도]` |
| `data.ctrl` | 모터 입력 | `[미는 힘 -1~1]` |

순서는 XML에 관절과 모터를 적은 순서다. 막대 각도 θ는 똑바로 서 있을 때 0이다.

### 강화학습 한 판의 흐름

```
리셋 → [관측 보기 → 행동 정하기 → 물리 진행 → 보상 받기 → 끝났나?] 반복
```

- **관측**: 정책이 보는 숫자. 각도 θ를 그대로 주지 않고 `cos θ`, `sin θ`로 준다. 각도는 한 바퀴 돌면 0과 2π가 같은 자세인데 숫자는 크게 다르다. cos/sin으로 바꾸면 그 문제가 없다. mjlab의 CartPole도 이렇게 한다.
- **보상**: 여러 항목의 가중합으로 만든다. 항목마다 이름과 가중치를 따로 두면 나중에 무엇이 학습을 이끄는지 보기 쉽다. 오리 학습 설정도 이 구조다.
- **종료**: 막대가 0.4 rad(약 23°) 넘게 기울거나 수레가 1.6m 넘게 벗어나면 "넘어짐"으로 끝낸다. 500 정책 스텝(10초)을 버티면 "시간 초과"로 끝낸다.
- **decimation**: 물리는 0.01초마다 계산하지만 정책은 0.02초마다 한 번 행동을 정한다. 정책 1스텝 = 물리 2스텝. 오리도 물리를 정책보다 촘촘하게 돌린다.

## 실습

`rl-tutorial/step1_random.py`를 만들고 `TODO`를 채운다.

```python
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("cartpole.xml")
data = mujoco.MjData(model)

print("관절 수 nq =", model.nq, "/ 모터 수 nu =", model.nu, "/ 시간 간격 =", model.opt.timestep)

DECIMATION = 2  # 정책 1번 = 물리 2스텝 (0.02초)


def observe(data):
    """정책이 볼 숫자 5개: [x, cos θ, sin θ, x 속도, θ 속도]"""
    # TODO 1: data.qpos, data.qvel에서 꺼내 np.float32 배열로 돌려준다
    ...


def reward_terms(data):
    """보상 항목별 값. 가중치는 아래 REWARD_WEIGHTS에서 곱한다."""
    x, theta = data.qpos
    _, theta_dot = data.qvel
    return {
        "alive": 1.0,                # 살아 있으면 매 스텝 +1
        "upright": np.cos(theta),    # 똑바로 서면 1, 기울수록 작아짐
        # TODO 2: 수레가 가운데(x=0)에서 멀수록 벌점. 힌트: x의 제곱에 마이너스
        "centered": ...,
        # TODO 3: 막대가 빨리 휘청일수록 벌점. 힌트: theta_dot의 제곱에 마이너스
        "calm": ...,
    }


REWARD_WEIGHTS = {"alive": 1.0, "upright": 1.0, "centered": 0.1, "calm": 0.01}


def reward(data):
    terms = reward_terms(data)
    return sum(REWARD_WEIGHTS[k] * v for k, v in terms.items())


def fell(data):
    # TODO 4: 막대가 0.4 rad 넘게 기울었거나 수레가 1.6 m 넘게 벗어나면 True
    ...


rng = np.random.default_rng(0)
lengths = []
for episode in range(5):
    mujoco.mj_resetData(model, data)                     # 처음 상태로
    data.qpos[:] = rng.uniform(-0.05, 0.05, size=2)      # 살짝 흔들어서 시작
    total, steps = 0.0, 0
    while True:
        data.ctrl[0] = rng.uniform(-1, 1)                # 무작위 행동
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)                  # 물리 0.01초 진행
        total += reward(data)
        steps += 1
        if fell(data) or steps >= 500:
            break
    lengths.append(steps)
    print(f"에피소드 {episode}: {steps}스텝 버팀, 보상 합 {total:.1f}, 마지막 관측 {observe(data).round(2)}")

print("평균 버틴 스텝:", np.mean(lengths))
```

실행:

```sh
uv run python step1_random.py
```

## 확인

이런 출력이 나오면 된다. 숫자는 조금 다를 수 있다.

```
관절 수 nq = 2 / 모터 수 nu = 1 / 시간 간격 = 0.01
에피소드 0: ...스텝 버팀, 보상 합 ..., 마지막 관측 [-0.08  0.91  0.41 -0.33  1.71]
...
평균 버틴 스텝: 30.2
```

- [ ] `nq = 2`, `nu = 1`이 나온다(관절 2개, 모터 1개).
- [ ] 무작위 행동은 평균 **30스텝 안팎(0.6초)** 만에 넘어진다. 이게 학습 전 실력이다. 2단계 목표는 500스텝이다.
- [ ] 마지막 관측의 `cos θ`가 0.92 근처다. cos 0.4 ≈ 0.92이니 0.4 rad에서 끝났다는 뜻이다.
- [ ] 마지막 관측의 4번째 값(수레 속도)이 0이 아니다. 계속 0이면 수레가 어딘가에 끼어 있다(0단계 "더 해 보기" 4번 참고).

## 더 해 보기

1. 무작위 행동 대신 `data.ctrl[0] = 0`(아무것도 안 함)이면 몇 스텝 버티는가?
2. 간단한 규칙을 손으로 짜 보자. "막대가 오른쪽으로 기울면 수레를 오른쪽으로 민다": `data.ctrl[0] = np.clip(5 * theta + 1 * theta_dot, -1, 1)`. 몇 스텝 버티는가? 숫자 5와 1을 바꿔 보자.
3. 2번처럼 사람이 짠 규칙이 이미 잘 된다면 강화학습은 왜 필요할까? 오리처럼 관절이 많은 로봇에서 이런 규칙을 손으로 짤 수 있을지 생각해 보자.

<details>
<summary>정답</summary>

```python
def observe(data):
    x, theta = data.qpos
    x_dot, theta_dot = data.qvel
    return np.array([x, np.cos(theta), np.sin(theta), x_dot, theta_dot], dtype=np.float32)


def reward_terms(data):
    x, theta = data.qpos
    _, theta_dot = data.qvel
    return {
        "alive": 1.0,
        "upright": np.cos(theta),
        "centered": -(x**2),
        "calm": -(theta_dot**2),
    }


def fell(data):
    x, theta = data.qpos
    return abs(theta) > 0.4 or abs(x) > 1.6
```

더 해 보기 답:
1. 평균 40스텝 정도로, 무작위와 비슷하게 금방 넘어진다. 막대는 원래 불안정하다.
2. 각도와 각속도에 비례해 미는 규칙(PD 제어)만으로 500스텝을 모두 버틴다. CartPole은 이렇게 손으로도 풀린다. 부호를 반대로(`-5 * theta ...`) 하면 오히려 더 빨리 넘어진다.
3. 오리는 관절이 많고, 걷기는 "어느 관절을 언제 얼마나"가 얽혀 있어서 규칙을 손으로 짜기 매우 어렵다. 강화학습은 보상만 정해 주면 그 규칙을 대신 찾는다.

</details>

## 오리와 연결

방금 짠 세 함수가 오리 학습 설정의 핵심이다. mjlab에서는 이렇게 이름이 바뀐다.

| 이 단계 | mjlab (4단계에서 확인) |
|---|---|
| `observe()` | `observations` 안의 `ObservationTermCfg` 여러 개 |
| `reward_terms()` + `REWARD_WEIGHTS` | `rewards` 안의 `RewardTermCfg(func=..., weight=...)` 여러 개 |
| `fell()`, `steps >= 500` | `terminations` 안의 `TerminationTermCfg` |
| `mj_resetData` + 무작위 시작 | `events` 안의 `mode="reset"` 항목 |
| `DECIMATION = 2` | `decimation` |

오리를 학습시킨다는 건 대부분 이 표의 오른쪽 칸을 고치는 일이다. 특히 보상 항목과 가중치를 고치는 데 시간이 가장 많이 든다.

[다음: 2단계 →](02-rsl-rl-ppo.md)
