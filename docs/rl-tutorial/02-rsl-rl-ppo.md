# 2단계: rsl_rl PPO로 학습하기

[← 1단계](01-mujoco-python.md) · [목차](README.md) · [다음: 3단계 →](03-onnx-play.md)

## 목표

- 1단계의 함수들을 **환경 64개를 한꺼번에 돌리는 클래스**로 묶는다.
- 오리 학습과 같은 학습기 **rsl_rl**의 PPO로 학습시킨다.
- TensorBoard로 학습 곡선을 보고, 막대를 500스텝 동안 세우게 만든다.

## 알아 둘 개념

### 왜 환경을 여러 개 돌리나

PPO는 "여러 판을 해 보고 → 결과를 모아 → 정책을 조금 고치기"를 반복한다. 한 번에 모으는 경험이 많을수록 학습이 안정적이다. 그래서 같은 환경을 여러 벌(여기서는 64개) 동시에 돌린다. 오리 학습은 GPU로 4096개를 돌린다. 이번에는 Python 반복문으로 64개를 차례로 돌려도 CPU로 충분히 빠르다.

### PPO 한 바퀴(iteration)

```
1. 수집: 64개 환경 × 24스텝 = 1536개 경험을 모은다 (정책은 행동에 약간의 무작위를 섞어 탐험)
2. 평가: 가치 함수(critic)로 "예상보다 잘했나 못했나(advantage)"를 계산한다
3. 개선: 잘한 행동은 더 자주, 못한 행동은 덜 하도록 정책을 조금 고친다 (너무 많이 바꾸지 않게 clip)
```

이걸 100바퀴 돌리면 15만 스텝 정도의 경험을 쓴다.

### rsl_rl이 환경에 요구하는 것 (`VecEnv`)

rsl_rl은 환경 클래스가 아래 모양이기만 하면 무엇이든 학습시킨다. mjlab도 결국 이 모양으로 감싸서 넘긴다.

| 속성·함수 | 뜻 |
|---|---|
| `num_envs`, `num_actions` | 환경 수, 행동 크기 |
| `max_episode_length` | 에피소드 최대 길이(정책 스텝) |
| `episode_length_buf` | 환경마다 지금 몇 스텝째인지 (torch 텐서) |
| `device`, `cfg` | 텐서가 있는 장치(`"cpu"`), 환경 설정(기록용) |
| `get_observations()` | 관측을 `TensorDict({"actor": ..., "critic": ...})`로 돌려준다 |
| `step(actions)` | 행동을 받아 `(관측, 보상, 끝남 여부, extras)`를 돌려준다. 끝난 환경은 **안에서 바로 리셋**한다 |

- 관측이 `actor`와 `critic` 두 묶음인 이유: 가치 함수는 학습 때만 쓰니까 정책보다 더 많은 정보(시뮬레이터만 아는 값)를 줘도 된다. 오리 학습이 실제로 그렇게 한다. 이번에는 둘 다 같은 값을 준다.
- `extras["time_outs"]`: 넘어져서 끝난 것과 시간이 다 돼서 끝난 것을 구분해 준다. 시간 초과는 "실패"가 아니므로 PPO가 다르게 처리한다.
- `extras["log"]`: `/`로 시작하는 이름으로 숫자를 넣으면 학습 로그와 TensorBoard에 같이 찍힌다. 보상 항목별 값을 넣어 두면 어느 항목이 학습을 이끄는지 보인다.

## 실습

### 1) 환경 클래스: `cartpole_env.py`

1단계 함수가 메서드로 들어갔다. `step()`의 `TODO`를 채운다.

```python
import mujoco
import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

REWARD_WEIGHTS = {"alive": 1.0, "upright": 1.0, "centered": 0.1, "calm": 0.01}


class CartPoleVecEnv(VecEnv):
    def __init__(self, num_envs=64, xml_path="cartpole.xml", max_episode_steps=500, device="cpu", seed=0):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.datas = [mujoco.MjData(self.model) for _ in range(num_envs)]  # 환경마다 상태 하나씩
        self.rng = np.random.default_rng(seed)

        # rsl_rl이 읽는 속성들
        self.num_envs = num_envs
        self.num_actions = self.model.nu
        self.max_episode_length = max_episode_steps
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.device = device
        self.cfg = {"reward_weights": REWARD_WEIGHTS}

        self.decimation = 2
        for i in range(num_envs):
            self._reset(i)

    # ---- 환경 하나 단위 (1단계 함수들) ----
    def _reset(self, i):
        d = self.datas[i]
        mujoco.mj_resetData(self.model, d)
        d.qpos[:] = self.rng.uniform(-0.05, 0.05, size=2)
        d.qvel[:] = self.rng.uniform(-0.05, 0.05, size=2)
        mujoco.mj_forward(self.model, d)  # 바꾼 qpos를 반영

    def _observe(self, d):
        x, theta = d.qpos
        x_dot, theta_dot = d.qvel
        return [x, np.cos(theta), np.sin(theta), x_dot, theta_dot]

    def _reward_terms(self, d):
        x, theta = d.qpos
        _, theta_dot = d.qvel
        return {"alive": 1.0, "upright": np.cos(theta), "centered": -(x**2), "calm": -(theta_dot**2)}

    def _fell(self, d):
        x, theta = d.qpos
        return abs(theta) > 0.4 or abs(x) > 1.6

    # ---- rsl_rl이 부르는 함수 ----
    def get_observations(self):
        obs = torch.tensor(np.array([self._observe(d) for d in self.datas]), dtype=torch.float32)
        return TensorDict({"actor": obs, "critic": obs}, batch_size=[self.num_envs])

    def step(self, actions):
        actions = actions.clamp(-1.0, 1.0).cpu().numpy()   # 모양: (num_envs, 1)
        rewards = torch.zeros(self.num_envs)
        dones = torch.zeros(self.num_envs, dtype=torch.long)
        time_outs = torch.zeros(self.num_envs, dtype=torch.long)
        term_sums = {k: 0.0 for k in REWARD_WEIGHTS}

        for i, d in enumerate(self.datas):
            # TODO 1: 행동을 모터 입력에 넣고 물리를 decimation번 진행

            # TODO 2: 보상 항목을 구해 가중합을 rewards[i]에 넣고,
            #         항목별 (가중치 × 값)을 term_sums에 더한다 (로그용)

            # TODO 3: 에피소드 길이를 1 늘리고, 넘어졌거나 시간 초과면
            #         dones[i]=1, (넘어지지 않고 시간 초과일 때만) time_outs[i]=1,
            #         그 환경을 리셋하고 길이를 0으로
            ...

        extras = {
            "time_outs": time_outs,
            "log": {f"/reward/{k}": v / self.num_envs for k, v in term_sums.items()},
        }
        return self.get_observations(), rewards, dones, extras
```

### 2) 학습 스크립트: `train.py`

설정은 mjlab의 CartPole 예제와 거의 같은 값이다. 그대로 쓴다.

```python
import sys
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from cartpole_env import CartPoleVecEnv

ITERATIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 100

train_cfg = {
    "num_steps_per_env": 24,          # 한 바퀴에 환경마다 모을 스텝 수
    "save_interval": 50,              # 몇 바퀴마다 체크포인트를 저장할지
    "obs_groups": {"actor": ["actor"], "critic": ["critic"]},
    "logger": "tensorboard",
    "actor": {                        # 정책 신경망
        "class_name": "MLPModel",
        "hidden_dims": [64, 64],
        "activation": "elu",
        "obs_normalization": False,
        # 행동에 섞을 무작위(탐험). 표준편차 1에서 시작해 학습하며 줄어든다
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    },
    "critic": {                       # 가치 함수 신경망
        "class_name": "MLPModel",
        "hidden_dims": [64, 64],
        "activation": "elu",
        "obs_normalization": False,
    },
    "algorithm": {
        "class_name": "PPO",
        "learning_rate": 1e-3,
        "gamma": 0.99,                # 할인율
        "lam": 0.95,                  # advantage 계산(GAE)용
        "clip_param": 0.2,            # 한 번에 정책을 얼마나 바꿀 수 있나
        "entropy_coef": 0.01,         # 탐험을 얼마나 유지하나
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "schedule": "adaptive",       # 정책이 너무 많이 바뀌면 학습률을 낮춤
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "rnd_cfg": None,
        "symmetry_cfg": None,
    },
}

env = CartPoleVecEnv(num_envs=64)
log_dir = f"logs/cartpole/{datetime.now():%Y-%m-%d_%H-%M-%S}"
runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device="cpu")
runner.learn(num_learning_iterations=ITERATIONS)
runner.export_policy_to_onnx(log_dir, "policy.onnx")   # 3단계에서 쓴다
print("저장 위치:", log_dir)
```

### 3) 먼저 3바퀴만 돌려 보기

큰 실행 전에는 항상 아주 작게 먼저 돌려서 오류가 없는지 본다.

```sh
uv run python train.py 3
```

### 4) 진짜 학습

```sh
uv run python train.py 100
```

M4 CPU에서 10초 안팎이면 끝난다. 화면에 바퀴마다 이런 표가 찍힌다.

```
                     Learning iteration 99/100
                            Total steps: 153600
                       Steps per second: ...
                            Mean reward: ...
                    Mean episode length: 500.00
                        Mean action std: ...
                          /reward/alive: 1.0000
                        /reward/upright: ...
```

### 5) TensorBoard로 보기

학습이 끝난 뒤(또는 학습 중에 다른 터미널에서):

```sh
uv run tensorboard --logdir logs
```

브라우저에서 http://localhost:6006 을 연다. 끝낼 때는 터미널에서 `Ctrl-C`.

## 확인

- [ ] `Mean episode length`가 20 안팎에서 시작해 100바퀴 안에 **500**에 닿는다. 확인할 때는 대략 19 → 125(20바퀴) → 226(40바퀴) → 441(80바퀴) → 500(100바퀴)으로 올랐다.
- [ ] `Mean action std`가 1에서 점점 줄어든다(탐험을 줄이고 배운 걸 쓰기 시작).
- [ ] TensorBoard의 `Train/mean_episode_length`와 `Train/mean_reward` 곡선이 위로 올라가고, `Policy/mean_std`는 내려간다. 보상 항목별 곡선은 `/reward/...`에 있다.
- [ ] `logs/cartpole/날짜/`에 `model_0.pt`, `model_50.pt`, `model_99.pt`, `policy.onnx`가 있다.

**학습이 안 되면** (에피소드 길이가 50 근처에서 안 올라가면) 이 순서로 의심한다.
1. 수레가 정말 움직이는가? 1단계 출력에서 수레 속도가 0이 아닌지 본다. 레일 충돌 같은 환경 문제가 가장 흔하다.
2. 행동이 모터에 들어가는가? `d.ctrl[0] = actions[i, 0]`이 빠지지 않았는지.
3. 리셋이 되는가? 넘어진 환경을 리셋하지 않으면 계속 넘어진 상태로 남는다.

## 더 해 보기

한 번에 하나씩 바꾸고 `train.py 100`을 다시 돌려 TensorBoard에서 곡선을 겹쳐 본다(실행마다 폴더가 따로 생긴다).

1. `REWARD_WEIGHTS`에서 `"alive"`를 0으로 하면? 버티는 것 자체의 보상이 사라져도 배우는가?
2. `"centered"`를 0으로 하면 수레가 한쪽으로 흘러가는지 3단계 재생에서 확인해 보자.
3. `num_envs`를 8로 줄이면? 같은 바퀴 수에서 학습이 어떻게 되는가? 256으로 늘리면?
4. `"init_std"`를 0.1로 줄여 탐험을 거의 안 하게 하면?
5. **일부러 꼼수 만들기**: `"alive"` 대신 `"upright"`만 크게(예: 10) 주고 종료 각도를 1.5 rad로 늘려 보자. 에이전트가 의도와 다른 방법으로 점수를 따는지 관찰한다. 이게 보상 해킹(reward hacking)이고, 오리 학습에서 가장 자주 마주치는 문제다.

<details>
<summary>정답 (step의 반복문 안)</summary>

```python
        for i, d in enumerate(self.datas):
            d.ctrl[0] = actions[i, 0]
            for _ in range(self.decimation):
                mujoco.mj_step(self.model, d)

            terms = self._reward_terms(d)
            rewards[i] = sum(REWARD_WEIGHTS[k] * v for k, v in terms.items())
            for k, v in terms.items():
                term_sums[k] += REWARD_WEIGHTS[k] * v

            self.episode_length_buf[i] += 1
            fell = self._fell(d)
            timeout = self.episode_length_buf[i] >= self.max_episode_length
            if fell or timeout:
                dones[i] = 1
                time_outs[i] = int(timeout and not fell)
                self._reset(i)
                self.episode_length_buf[i] = 0
```

넘어진 환경을 리셋한 뒤 돌려주는 관측은 **리셋된 새 판의 첫 관측**이다. rsl_rl은 `dones`를 보고 판이 바뀐 걸 안다.

</details>

## 오리와 연결

| 이 단계 | 오리 학습 |
|---|---|
| `CartPoleVecEnv` (Python 반복문, 64개) | mjlab `ManagerBasedRlEnv` (GPU, 4096개)를 `RslRlVecEnvWrapper`로 감싼 것 |
| `train_cfg` 딕셔너리 | mjlab의 `RslRlOnPolicyRunnerCfg` (같은 항목을 dataclass로 적은 것) |
| `OnPolicyRunner(...).learn(...)` | `colab-duck train`이 VM에서 실제로 부르는 것 |
| `logs/cartpole/날짜/model_*.pt` | `logs/rsl_rl/실험이름/날짜/model_*.pt` (`pull`로 받는 것) |
| `tensorboard --logdir logs` | `duck-local tb` |

오리 학습 로그에 찍히는 `Mean episode length`, `Mean action std`, 보상 항목별 값도 지금 본 것과 같은 뜻이다.

[다음: 3단계 →](03-onnx-play.md)
