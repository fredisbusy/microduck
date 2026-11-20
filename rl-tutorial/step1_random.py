import mujoco
import numpy as np

# [환경] 물리 세계: model은 규칙(모양·질량·관절), data는 지금 이 순간의 상태
model = mujoco.MjModel.from_xml_path("cartpole.xml")
data = mujoco.MjData(model)
"""
data.qpos: 관절 위치 [수레위치, 막대각도]
data.qvel: 관절 속도 [수레속도, 막대각속도(각속도: 각도가 1초에 얼마나 변하는지)]
data.ctrl: 모터 입력 [미는 힘]

ẋ ("x dot") = dx/dt = 위치의 변화율 = 속도
θ̇ ("theta dot") = dθ/dt = 각도의 변화율 = 각속도
ẍ ("x double dot") = 가속도
"""

print(
    "관절 수 nq =",
    model.nq,
    "/ 모터 수 nu =",
    model.nu,
    r"\ 시간 간격 =",
    model.opt.timestep,
)

# [환경] 행동 1번에 물리를 몇 번 진행할지 (0.01초 × 2 = 0.02초)
DECIMATION = 2
# [보상] 항목별 가중치
REWARD_WEIGHTS = {
    "alive": 1.0,
    "upright": 1.0,
    "centered": 0.1,
    "calm": 0.01,
}


# [상태] 환경의 상태(data)에서 정책이 볼 숫자만 뽑은 것 = 관측
def observe(data):
    """정책이 볼 숫자 5개: [x, cos θ, sin θ, x 속도, θ 속도]"""
    # TODO 1: data.qpos, data.qvel에서 꺼내 np.float32 배열로 돌려준다
    x, theta = data.qpos
    x_dot, theta_dot = data.qvel
    return np.array(
        [x, np.cos(theta), np.sin(theta), x_dot, theta_dot], dtype=np.float32
    )


# [보상] 지금 상태가 얼마나 좋은지 항목별로 매긴 점수
def reward_items(data):
    """보상 항목별 값. 가중치는 아래 REWARD_WEIGHTS에서 곱한다."""
    x, theta = data.qpos
    _, theta_dot = data.qvel
    return {
        "alive": 1.0,  # 살아 있으면 매 스텝 +1
        "upright": np.cos(theta),  # 똑바로 서면 1, 기울수록 작아짐
        # TODO 2: 수레가 가운데(x=0)에서 멀수록 벌점. 힌트: x의 제곱에 마이너스
        "centered": -(x**2),
        # TODO 3: 막대가 빨리 휘청일수록 벌점. 힌트: theta_dot의 제곱에 마이너스
        "calm": -(theta_dot**2),
    }


# [보상] 항목 점수 × 가중치를 더한 한 스텝의 보상
def reward(data):
    terms = reward_items(data)
    return sum(REWARD_WEIGHTS[k] * v for k, v in terms.items())


# [환경] 에피소드가 끝나는 조건 (넘어짐 판정)
def fell(data):
    # TODO 4: 막대가 0.4 rad 넘게 기울었거나 수레가 1.6 m 넘게 벗어나면 True
    x, theta = data.qpos
    return bool(np.abs(theta) > 0.4 or np.abs(x) > 1.6)


rng = np.random.default_rng(0)
lengths = []

for episode in range(100):
    # [환경] 리셋: 상태를 처음으로 되돌리고 살짝 흔들어서 시작
    mujoco.mj_resetData(model, data)
    data.qpos[:] = rng.uniform([-0.05, -0.05], [0.05, 0.05], size=2)
    total, steps = 0.0, 0
    while True:
        # [정책] 관측을 보고 행동을 고르는 부분. 지금은 관측을 안 보고 무작위로 고른다
        # [행동] 고른 값(-1~1)을 모터 입력 data.ctrl에 넣는다
        data.ctrl[0] = rng.uniform(-1, 1)
        # [환경] 행동을 받아 물리를 진행해 다음 상태로 넘어간다
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)
        # [보상] 바뀐 상태에 점수를 매겨 누적
        total += reward(data)
        steps += 1
        # [환경] 넘어졌거나 500스텝(10초)을 채우면 에피소드 끝
        if fell(data) or steps >= 500:
            break
    lengths.append(steps)
    print(
        f"에피소드 {episode}: {steps}스텝 버팀, 보상 합 {total:.1f}, 마지막 관측 {observe(data).round(2)}"
    )

print("평균 버틴 스텝:", np.mean(lengths))
