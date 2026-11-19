# microduck workspace

Pollen Robotics Microduck(오픈소스 이족보행 오리 로봇)을 **장비 없이** 만져보는 작업 폴더.
공식/커뮤니티 저장소는 그대로 클론해 두고(이 저장소에서 추적 안 함), 내 노트북과 메모만 여기서 관리한다.

| 폴더 | 정체 | 용도 |
|---|---|---|
| `microduck/` | [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck) (main + 로컬 브랜치 `mac-local`) | 로봇 소프트웨어(Rust). `scripts/duck-sim`으로 MuJoCo 위에서 실제 데몬 실행 |
| `microduck_rl/` | [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) (develop) | 공식 RL 학습(mjlab, CUDA 필요) → Colab에서 학습. Mac에선 정책 재생·duck-sim 시뮬레이터 |
| `microduck-lab/` | [jonathanhawkins/microduck-lab](https://github.com/jonathanhawkins/microduck-lab) (비공식) | Mac CPU로 학습 + 브라우저 뷰어. 안에 공식 저장소 2개를 **고정 sha**로 따로 클론해 둠 |
| `notebooks/microduck_rl_colab.ipynb` | 내 것 | Colab GPU로 공식 학습 → Drive에 체크포인트/ONNX |

## 처음 받았을 때

위 세 저장소는 이 저장소에 포함되지 않으므로 직접 클론한다 (macOS, Apple Silicon 기준).

```sh
git clone https://github.com/fredisbusy/microduck && cd microduck
git clone https://github.com/pollen-robotics/microduck
git clone https://github.com/pollen-robotics/microduck_rl
git clone https://github.com/jonathanhawkins/microduck-lab

# macOS 빌드 패치 (아래 "로컬 패치" 참고)
git -C microduck switch -c mac-local && git -C microduck apply ../patches/microduck-configd-macos-clock.patch

# microduck_rl: duck-sim이 쓰는 venv
(cd microduck_rl && uv sync)

# microduck-lab: 공식 저장소를 고정 sha로 lab 안에 클론 (sha는 microduck-lab/AGENTS.md "Setup")
cd microduck-lab
git clone https://github.com/pollen-robotics/microduck_rl && git -C microduck_rl checkout badc4e7ffe5507fd7acb1a21487bd2925c1afe5a
git clone https://github.com/pollen-robotics/microduck && git -C microduck checkout 2c61dcc1f03440541cdc0729f7a375b2a9ea3005
(cd microduck_local && uv sync) && (cd duck-viewer && npm install)
```

## 준비물 (설치됨)

- uv, node, git
- Rust: `brew install rustup` → `rustup default stable`. 셸에서 쓰려면 `~/.zshrc`에
  `export PATH="/opt/homebrew/opt/rustup/bin:$PATH"` 추가 필요 (아직 안 함).

## 1. Mac에서 학습 (microduck-lab)

```sh
cd microduck-lab/microduck_local
uv run train-walk --envs 32 --steps 3_000_000 --run-name first-gait   # M1 Pro ~5분
uv run eval-walk runs/first-gait/policy.onnx
uv run duck-lab runs/first-gait ../microduck/policies/alpha_walking.onnx
cd ../duck-viewer && npm run dev      # 출력된 URL 열기 (/ 랩, /sim 월드, /train 차트)
```

실제 로봇에 올리지 말 것(도메인 랜덤화 일부만 적용). 실험·보상 설계용.

## 2. 로봇 소프트웨어 시뮬레이션 (duck-sim)

```sh
cd microduck
export PATH="/opt/homebrew/opt/rustup/bin:$PATH" DUCK_SIM_RL="$HOME/Projects/microduck/microduck_rl"
scripts/duck-sim                 # MuJoCo 창이 뜨고 오리가 일어섬 (창 없이: DUCK_SIM_VIEWER=0)
scripts/duck-sim drive 0.3 0     # vx m/s, vyaw rad/s 로 8초 걷기
scripts/duck-sim ctl robot do roulade
scripts/duck-sim down
```

- 일어선 직후 첫 `drive`는 안 걷는 경우가 있음 → 한 번 더.
- 측정(2026-11-19): 0.15 m/s 요청 → 0.086, 0.3 → 0.139. 제자리 회전은 거의 안 됨(업스트림 문서상 미해결).

### 로컬 패치 (`microduck` 의 `mac-local` 브랜치, `patches/microduck-configd-macos-clock.patch`)

main(0.23.0)의 `configd/src/wifi_ssh.rs`가 Linux 전용 `CLOCK_BOOTTIME`을 써서 macOS 빌드 실패 →
macOS에선 `CLOCK_MONOTONIC`(수면 시간 포함)을 쓰도록 cfg 분기. 업스트림이 고치면 브랜치 버리고 main으로.

```sh
git -C microduck switch main && git -C microduck pull && git -C microduck rebase main mac-local
```

## 3. Colab에서 공식 학습

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fredisbusy/microduck/blob/main/notebooks/microduck_rl_colab.ipynb)

강화학습 입문자용 Getting Started 노트북: 개념 설명 → 맛보기 학습(100회)과 "아기 오리" 영상 → 본 학습 → 성장 과정 영상 비교 → TensorBoard 읽는 법 → ONNX 내보내기.
GPU 런타임으로 위에서부터 실행. 체크포인트·영상·ONNX는 `MyDrive/microduck/`에 저장. Mac에서 확인:

```sh
cd microduck_rl && uv run scripts/infer_policy.py --walking <받은.onnx>
```

## 업데이트

```sh
git -C microduck_rl pull && (cd microduck_rl && uv sync)
git -C microduck-lab pull          # lab 안의 고정 sha 클론은 AGENTS.md의 sha가 바뀌면 그에 맞춰 checkout
```
