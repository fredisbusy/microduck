# 3단계: ONNX로 내보내 재생하기

[← 2단계](02-rsl-rl-ppo.md) · [목차](README.md) · [다음: 4단계 →](04-mjlab.md)

## 목표

- 2단계에서 만든 `policy.onnx`를 열어 입력과 출력 모양을 확인한다.
- PyTorch와 rsl_rl 없이 **onnxruntime만으로** 정책을 돌려 뷰어에서 막대가 서는 걸 본다.

## 알아 둘 개념

### 왜 ONNX로 내보내나

학습할 때는 PyTorch, rsl_rl, 가치 함수, 탐험용 무작위까지 다 필요하다. 하지만 로봇에서 정책을 돌릴 때는 **"관측 → 행동" 계산 하나**만 있으면 된다. ONNX는 그 계산만 떼어 낸 파일 형식이다. 작고, 여러 언어(Python, C++, Rust)에서 같은 결과로 돌릴 수 있다. 오리 로봇 소프트웨어(Rust)도 ONNX 정책 파일을 읽어서 걷는다.

내보낼 때는 탐험용 무작위가 빠진다. 학습 중에는 행동에 무작위를 섞었지만, 재생할 때는 정책이 가장 좋다고 보는 행동(평균값)을 그대로 쓴다.

### 가장 흔한 실수: 관측 순서

ONNX 파일 안에는 "5개 숫자를 받는다"는 것만 있고, 각 숫자가 무슨 뜻인지는 없다. 재생 코드가 학습 때와 **같은 순서, 같은 단위**로 관측을 만들어야 한다. 순서를 하나만 바꿔도 정책이 엉뚱하게 움직인다. 실제 로봇에 정책을 올릴 때 가장 많이 나는 사고다.

## 실습

### 1) ONNX 파일 들여다보기

```sh
ls logs/cartpole/*/
```

가장 최근 폴더에 `policy.onnx`와 `policy.onnx.data`가 있다. `.data`는 신경망 가중치가 따로 저장된 파일이니 **둘을 항상 같이 옮긴다.**

### 2) 재생 스크립트: `play.py`

`TODO`를 채운다.

```python
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

onnx_path = sys.argv[1]
headless = "--headless" in sys.argv   # 창 없이 숫자로만 확인할 때

session = ort.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name
print("ONNX 입력:", [(i.name, i.shape) for i in session.get_inputs()])
print("ONNX 출력:", [(o.name, o.shape) for o in session.get_outputs()])

model = mujoco.MjModel.from_xml_path("cartpole.xml")
data = mujoco.MjData(model)
DECIMATION = 2


def observe(d):
    # TODO 1: 2단계 _observe와 같은 순서로, 모양 (1, 5)의 float32 배열
    ...


def policy_step():
    # TODO 2: session.run(None, {input_name: 관측})[0] 으로 행동을 얻고,
    #         -1~1로 자른 뒤 data.ctrl[0]에 넣고, 물리를 DECIMATION번 진행
    ...


data.qpos[1] = 0.1  # 막대를 살짝 기울인 채로 시작

if headless:
    for _ in range(1000):  # 1000 정책 스텝 = 20초
        policy_step()
    print(f"20초 뒤 카트 위치 {data.qpos[0]:.3f}, 막대 각도 {data.qpos[1]:.3f} rad")
else:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.time()
            policy_step()
            viewer.sync()
            # 실제 시간과 맞추기: 정책 한 스텝(0.02초)만큼 기다림
            time.sleep(max(0, model.opt.timestep * DECIMATION - (time.time() - start)))
```

### 3) 숫자로 먼저 확인

`날짜` 부분은 실제 폴더 이름으로 바꾼다(`Tab` 자동 완성).

```sh
uv run python play.py logs/cartpole/날짜/policy.onnx --headless
```

### 4) 뷰어로 보기

Mac에서 Python 코드 안에서 뷰어 창을 띄울 때는 `python` 대신 **`mjpython`** 을 쓴다.

```sh
uv run mjpython play.py logs/cartpole/날짜/policy.onnx
```

뷰어에서 막대를 더블클릭하고 `Ctrl` + 오른쪽 드래그로 툭 쳐 보자. 정책이 수레를 움직여 다시 세우는지 본다.

## 확인

- [ ] 입력이 `('obs', [1, 5])`, 출력이 `('actions', [1, 1])`로 찍힌다.
- [ ] `--headless`로 돌리면 20초 뒤 막대 각도가 0 근처다. 확인할 때는 `카트 위치 -0.113, 막대 각도 0.000 rad`이었다.
- [ ] 뷰어에서 막대가 쓰러지지 않고, 손으로 쳐도 되살린다.

## 더 해 보기

1. `observe()`에서 `cos θ`와 `sin θ`의 순서를 바꿔 보자. 무슨 일이 생기는가? (관측 순서 실수 체험)
2. 2단계에서 `model_0.pt`만 있을 때(학습 0바퀴)의 정책도 보고 싶다면 `train.py 1`로 학습한 뒤 그 폴더의 ONNX를 재생해 학습 전후를 비교한다.
3. 막대를 세게 쳐서 0.4 rad 넘게 기울여 보자. 학습 때 그 각도까지는 겪어 본 적이 없다(넘어진 걸로 끝났으니까). 정책이 처음 보는 상황에서 어떻게 행동하는가?
4. 2단계 "더 해 보기"에서 `centered` 가중치를 0으로 학습한 정책을 재생해 보고, 수레가 레일 끝으로 흘러가는지 본다.

3번이 실제 로봇 문제와 이어진다. 시뮬레이터에서 겪어 보지 않은 상황(다른 바닥, 다른 무게, 밀침)에서 정책이 무너지지 않게 하려고, 오리 학습은 학습 중에 일부러 무게·마찰을 바꾸고 몸을 밀어 준다(도메인 랜덤화).

<details>
<summary>정답</summary>

```python
def observe(d):
    x, theta = d.qpos
    x_dot, theta_dot = d.qvel
    return np.array([[x, np.cos(theta), np.sin(theta), x_dot, theta_dot]], dtype=np.float32)


def policy_step():
    action = session.run(None, {input_name: observe(data)})[0]
    data.ctrl[0] = np.clip(action[0, 0], -1, 1)
    for _ in range(DECIMATION):
        mujoco.mj_step(model, data)
```

- 대괄호가 두 겹(`[[...]]`)인 이유: ONNX 입력 모양이 `(1, 5)`, 즉 "환경 1개 × 관측 5개"다.
- 더 해 보기 1번: 정책이 엉뚱한 방향으로 밀어서 금방 쓰러진다.

</details>

## 오리와 연결

| 이 단계 | 오리 |
|---|---|
| `runner.export_policy_to_onnx(...)` | `colab-duck export`, 노트북의 ONNX 내보내기 |
| `play.py` (onnxruntime + MuJoCo 뷰어) | `duck-sim`: 로봇 소프트웨어가 ONNX 정책을 읽어 MuJoCo 위에서 걷기 |
| 관측 순서를 학습과 똑같이 맞추기 | 로봇 소프트웨어의 관측 코드가 학습 설정과 같은 순서여야 하는 이유 |

[다음: 4단계 →](04-mjlab.md)
