# 4단계: mjlab으로 다시 만들기

[← 3단계](03-onnx-play.md) · [목차](README.md)

## 목표

- 오리 학습이 쓰는 틀인 **mjlab**을 설치하고, mjlab에 들어 있는 CartPole 예제를 Mac CPU로 작게 학습시킨다.
- 브라우저 3D 뷰어 **viser**로 재생한다.
- mjlab의 CartPole 설정 파일을 읽고, 0~3단계에서 손으로 만든 부품과 하나씩 짝지어 본다. **이 짝짓기가 튜토리얼의 마지막 과제다.**

## 알아 둘 개념

### mjlab이 해 주는 것

0~3단계에서 직접 짠 것들(환경 여러 벌, 관측·보상·종료 함수, 리셋, rsl_rl 연결, 로그)을 mjlab은 **설정 조각을 조립하는 방식**으로 해 준다. 보상 하나를 더하려면 함수 하나를 쓰고 `RewardTermCfg`로 등록하면 된다. 나머지는 mjlab이 알아서 묶는다. 그리고 물리를 GPU용 MuJoCo(mujoco_warp)로 돌려서 환경 수천 개를 한꺼번에 돌린다.

### Mac CPU에서는 느리다

mujoco_warp는 GPU용 엔진이라 CPU에서는 한 코어로 차례차례 돈다. 오리 학습을 이 Mac에서 재 보니 초당 400~800스텝이었다. 2단계 직접 만든 환경(초당 약 2만 스텝)보다 훨씬 느리다. 그래서 이 단계는 **반드시 작게** 돌린다. 처음 실행할 때는 커널 컴파일 때문에 시작까지 시간이 더 걸린다.

## 실습

### 1) 설치

```sh
uv add "mjlab==1.3.0"
```

rsl_rl 5.0.1 등 지금까지 쓴 라이브러리와 판이 맞는다. 설치가 꽤 크다(warp, viser, wandb 등).

설치된 예제 목록 확인:

```sh
uv run list-envs
```

`Mjlab-Cartpole-Balance`(서 있는 막대 유지)와 `Mjlab-Cartpole-Swingup`(아래로 늘어진 막대를 휘둘러 세우기)이 보이면 된다.

### 2) 학습 전에 환경부터 보기

1단계에서 무작위 행동으로 학습 전 실력을 쟀던 것처럼, mjlab에도 그 기능이 있다.

```sh
uv run play Mjlab-Cartpole-Balance --agent random --viewer viser --device cpu
```

터미널에 찍히는 주소(예: `http://localhost:8080`)를 브라우저로 연다. 다 보면 터미널에서 `Ctrl-C`.

### 3) 3바퀴만 돌려 보기

```sh
uv run train Mjlab-Cartpole-Balance \
  --gpu-ids None \
  --env.scene.num-envs 64 \
  --agent.max-iterations 3 \
  --agent.logger tensorboard
```

| 옵션 | 뜻 |
|---|---|
| `--gpu-ids None` | GPU 없이 CPU로. **Mac에서는 꼭 필요하다.** 없으면 GPU를 찾다가 `IndexError`로 죽는다 |
| `--env.scene.num-envs 64` | 환경 수. 예제 기본값은 1이다 |
| `--agent.max-iterations 3` | PPO 바퀴 수 |
| `--agent.logger tensorboard` | 기본값은 wandb(온라인 서비스)다. 계정 없이 쓰려면 tensorboard로 |

출력의 `Steps per second`를 보고 100바퀴(64 × 32 × 100 ≈ 20만 스텝)에 몇 분 걸릴지 계산해 본다. 초당 800스텝이면 약 4분, 400스텝이면 약 8분이다. 이 숫자는 오리 학습 측정값에서 나온 추정이고, CartPole로 직접 재 보지는 않았다.

### 4) 학습

계산한 시간이 괜찮으면:

```sh
uv run train Mjlab-Cartpole-Balance \
  --gpu-ids None \
  --env.scene.num-envs 64 \
  --agent.max-iterations 100 \
  --agent.logger tensorboard
```

결과는 `logs/rsl_rl/cartpole/날짜/`에 쌓인다. 2단계와 같은 `logs` 폴더라 TensorBoard 하나로 내 CartPole과 mjlab CartPole을 같이 볼 수 있다.

```sh
uv run tensorboard --logdir logs
```

### 5) 재생

```sh
uv run play Mjlab-Cartpole-Balance \
  --checkpoint-file logs/rsl_rl/cartpole/날짜/model_99.pt \
  --viewer viser --device cpu
```

`--agent random`일 때와 비교해 막대가 서 있으면 성공이다.

### 6) 마지막 과제: 설정 파일 읽고 짝짓기

mjlab CartPole 설정 파일을 연다.

```sh
uv run python -c "import mjlab.tasks.cartpole as m, os; print(os.path.dirname(m.__file__))"
```

출력된 폴더의 `cartpole_env_cfg.py`와 `cartpole.xml`을 VS Code로 열고, 아래 표의 빈칸을 채운다. 오른쪽 칸에는 mjlab 파일에서 찾은 이름이나 값을 적는다.

| 0~3단계에서 내가 만든 것 | mjlab에서는? |
|---|---|
| `cartpole.xml`의 `slider`, `hinge` | |
| `push` 모터, `gear="10"` | |
| `observe()`의 5개 값 | `observations["actor"]` 안의 항목 4개: |
| `cos θ`, `sin θ` 만드는 코드 | |
| `reward_terms()` + `REWARD_WEIGHTS` | |
| `fell()` (넘어지면 끝) | |
| `steps >= 500` (시간 초과) | |
| `_reset()`의 무작위 시작 | |
| `DECIMATION = 2`, `timestep=0.01` | |
| `train_cfg`의 `hidden_dims`, `learning_rate` 등 | |
| `train_cfg`의 `num_steps_per_env = 24` | |

다 채웠으면 아래 질문에 답해 본다.

1. mjlab CartPole에는 `fell()`에 해당하는 종료 조건이 있는가? 없다면 막대가 쓰러진 뒤에는 어떻게 되고, 그걸 보상이 어떻게 처리하는가?
2. mjlab의 보상은 항목 하나(`smooth_reward`)인데 안에서 네 값을 **곱한다.** 내 보상은 네 항목을 **더했다.** 곱하면 무엇이 달라지는가? (힌트: 막대가 쓰러진 상태라면?)
3. `observations["actor"]`에만 있는 `enable_corruption=True`는 무엇을 하는 것 같은가? 왜 `critic`에는 없는가?

<details>
<summary>짝짓기 정답</summary>

| 0~3단계에서 내가 만든 것 | mjlab에서는 |
|---|---|
| `cartpole.xml`의 `slider`, `hinge` | mjlab `cartpole.xml`의 `slider`, `hinge_1` (거의 같은 XML. 레일이 두 줄이고 클래스로 기본값을 묶었다) |
| `push` 모터, `gear="10"` | `<motor name="slide" ... gear="10">`, 파이썬에서는 `XmlActuatorCfg`로 XML 모터를 그대로 쓰고 `JointEffortActionCfg`(`actions["effort"]`)로 행동을 연결 |
| `observe()`의 5개 값 | `cart_pos`, `pole_angle`(2개 값), `cart_vel`, `pole_vel` → 합쳐서 5개 |
| `cos θ`, `sin θ` 만드는 코드 | `pole_angle_cos_sin()` 함수 |
| `reward_terms()` + `REWARD_WEIGHTS` | `rewards["smooth_reward"] = RewardTermCfg(func=cartpole_smooth_reward, weight=1.0)` |
| `fell()` | 없음 (Balance 예제는 넘어져도 끝내지 않는다) |
| `steps >= 500` | `terminations["time_out"]` + `episode_length_s=50.0` |
| `_reset()`의 무작위 시작 | `events["reset_slider"]`, `events["reset_hinge"]` (`mode="reset"`, `reset_joints_by_offset`) |
| `DECIMATION = 2`, `timestep=0.01` | `decimation=5`, `MujocoCfg(timestep=0.01)` → 정책은 0.05초마다 |
| `hidden_dims`, `learning_rate` 등 | `cartpole_ppo_runner_cfg()`의 `RslRlModelCfg`, `RslRlPpoAlgorithmCfg` (값이 거의 같다) |
| `num_steps_per_env = 24` | `num_steps_per_env=32` |

질문 답:
1. 없다. 쓰러져도 50초까지 계속 간다. 대신 보상의 `upright = (cos θ + 1) / 2`가 쓰러진 동안 낮게 나와서, 쓰러지면 손해라는 걸 배운다.
2. 곱하면 한 항목이라도 0에 가까우면 전체가 작아진다. 막대가 쓰러지면(upright ≈ 0) 수레가 가운데 있든 말든 보상이 거의 없다. "모든 조건을 동시에 만족해야 점수"라는 뜻이다. 더하면 하나를 버리고 다른 걸로 점수를 메우는 꼼수가 생기기 쉽다.
3. 정책이 보는 관측에 잡음을 섞는다. 실제 로봇 센서는 잡음이 있으니 정책이 잡음에 강해지도록 하는 것이다. critic은 학습 때만 쓰고 로봇에 올라가지 않으니 깨끗한 값을 준다.

</details>

## 오리와 연결: 이제 오리 설정을 열어 보자

여기까지 왔으면 오리 학습 설정을 읽을 준비가 됐다. 공식 저장소를 받아 둔 상태라면(`README.md`의 "처음 받았을 때"):

```sh
cd ~/Projects/microduck/microduck_rl
grep -rn "RewardTermCfg\|ObservationTermCfg\|EventTermCfg\|TerminationTermCfg" --include=*.py . | head -50
```

CartPole에서 본 것과 같은 조각들이 훨씬 많이 나온다. 이렇게 읽어 보면 된다. 아래 항목 예시는 걷는 로봇 학습에서 흔히 쓰는 것들이고, 오리 설정에 실제로 무엇이 있는지는 직접 열어 확인한다.

1. **XML**: 오리 몸, 다리 관절, 모터
2. **observations**: 몸 기울기, 관절 각도·속도, 이전 행동, 걷기 명령(앞으로 얼마나 빨리) 등
3. **rewards**: 명령한 속도 따라가기, 몸 똑바로, 발 미끄러짐 벌점, 에너지 벌점 등 여러 항목과 가중치
4. **events**: 리셋 외에 마찰·무게 바꾸기, 몸 밀기 같은 도메인 랜덤화
5. **runner 설정**: 신경망 크기, PPO 값, 환경 수 4096, 반복 수천 번

CartPole에서는 각 조각이 한두 개였고, 오리에서는 열 개 넘게 있다는 것이 차이다. 구조는 같다.

## 다음에 해 볼 만한 것

- **Swingup**: `Mjlab-Cartpole-Swingup`은 아래로 늘어진 막대를 휘둘러 세워야 한다. 훨씬 어렵다. CPU로 오래 걸리면 Colab이나 HF Jobs로 보낸다.
- **내 CartPole을 mjlab 작업으로 등록하기**: `cartpole_env_cfg.py`를 복사해 내 XML과 내 보상(합 방식)으로 바꾸고 `register_mjlab_task`로 등록한 뒤 학습시킨다.
- **오리 보상 하나 바꿔 보기**: 오리 보상 가중치 하나를 바꾸고 Mac에서 64개 환경으로 몇 바퀴만 돌려 오류가 없는지 본 뒤(스모크 테스트), 본 학습은 Colab·HF Jobs로 보낸다.

[← 목차로](README.md)
