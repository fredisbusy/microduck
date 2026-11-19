# microduck workspace

Pollen Robotics Microduck(오픈소스 이족보행 오리 로봇)을 **장비 없이** 만져보는 작업 폴더.
공식/커뮤니티 저장소는 그대로 클론해 두고(이 저장소에서 추적 안 함), 내 노트북과 메모만 여기서 관리한다.

| 폴더 | 정체 | 용도 |
|---|---|---|
| `microduck/` | [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck) (main + 로컬 브랜치 `mac-local`) | 로봇 소프트웨어(Rust). `scripts/duck-sim`으로 MuJoCo 위에서 실제 데몬 실행 |
| `microduck_rl/` | [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) (develop) | 공식 RL 학습(mjlab, CUDA 필요) → Colab·Kaggle에서 학습. Mac에선 정책 재생·duck-sim 시뮬레이터 |
| `microduck-lab/` | [jonathanhawkins/microduck-lab](https://github.com/jonathanhawkins/microduck-lab) (비공식) | Mac CPU로 학습 + 브라우저 뷰어. 안에 공식 저장소 2개를 **고정 sha**로 따로 클론해 둠 |
| `notebooks/microduck_rl_colab.ipynb` | 내 것 | Colab·Kaggle GPU로 공식 학습 → HF Hub(`<아이디>/microduck-runs`)에 체크포인트/ONNX |

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

## 3. Colab·Kaggle에서 공식 학습

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fredisbusy/microduck/blob/main/notebooks/microduck_rl_colab.ipynb)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/fredisbusy/microduck/blob/main/notebooks/microduck_rl_colab.ipynb)

강화학습 입문자용 Getting Started 노트북: 개념 설명 → 맛보기 학습(100회)과 "아기 오리" 보기 → 본 학습 → 성장 과정 비교 → TensorBoard 읽는 법 → ONNX 내보내기.
오리는 Mac에서 3D 뷰어(`scripts/duck-local view`, mjlab의 viser)로 본다. 노트북은 학습과 ONNX 내보내기만.

**Colab과 Kaggle을 번갈아 쓸 수 있다.** 노트북이 실행 환경을 알아서 판단하고, 결과는 두 곳 모두 Hugging Face Hub의 비공개 저장소 `<아이디>/microduck-runs`에 올린다(학습 중 1분마다). 한쪽 GPU 한도가 차면 다른 쪽에서 1-2, 1-3 → 3-2로 이어서 학습한다.

- 처음 한 번: HF **Write** 토큰을 만들어 노트북 비밀값 `HF_TOKEN`으로 넣기 (Colab: 🔑 보안 비밀 + 노트북 액세스, Kaggle: Add-ons → Secrets).
- Kaggle: Settings에서 Accelerator **GPU T4 x2**(학습은 1장만 씀), **Internet On**. 전화번호 인증 필요. 한 세션 최대 12시간.
- Mac에서 처음 한 번 `cd microduck_rl && uv run hf auth login`.

Mac에서 확인:

```sh
scripts/duck-local pull                 # HF 저장소 → runs/hf (바뀐 것만)
scripts/duck-local view main            # 3D 뷰어 http://localhost:8090
scripts/duck-local tb                   # TensorBoard http://localhost:6007
cd microduck_rl && uv run scripts/infer_policy.py --walking ../runs/hf/exports/walking_iter2000.onnx
```

## 4. VS Code + Claude Code로 Colab GPU 쓰기

사람은 VS Code에서, Claude는 터미널에서 **같은 Colab GPU 런타임**을 쓴다. Colab VS Code 확장은 런타임을 에이전트에게 열어 주지 않는다는 관리자 답변이 있어서([colab-vscode#433](https://github.com/googlecolab/colab-vscode/issues/433)), Google 공식 [google-colab-cli](https://github.com/googlecolab/google-colab-cli)를 쓴다.

처음 한 번:

```sh
uv tool install google-colab-cli
colab sessions        # 출력된 URL을 브라우저에서 승인 → 받은 코드를 붙여넣기 (Google 로그인)
```

`~/.ssh/config`에 `Host colab-duck`을 둔다(이 Mac에는 설정돼 있음. `Host *`의 1Password agent보다 위에 있어야 함):

```
Host colab-duck
    HostName colab-duck
    User root
    ProxyCommand ~/.local/bin/colab ssh --proxy-mode -s duck --gpu T4   # 실제로는 절대 경로
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    IdentityAgent none
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ControlMaster auto
    ControlPath ~/.ssh/cm-colab-duck
    ControlPersist 10m
```

매번 (`scripts/colab-duck help`, VS Code에서는 `Tasks: Run Task` → `Colab: ...`):

```sh
scripts/colab-duck up               # T4 런타임 + 로컬 microduck_rl 올리기 + uv sync
scripts/colab-duck train quick 1024 100
scripts/colab-duck logs quick
scripts/colab-duck view quick       # http://localhost:8080 3D 뷰어 (Ctrl-C 종료)
scripts/colab-duck tb               # http://localhost:6006 TensorBoard
scripts/colab-duck pull             # 체크포인트 → runs/colab/
scripts/colab-duck export main      # ONNX → runs/colab/
scripts/colab-duck down             # 꼭 끄기 (켜 두면 컴퓨팅 단위가 계속 나감)
```

- 로컬 `microduck_rl/`이 기준이다. 보상 등을 고치면 `scripts/colab-duck sync`.
- 폰 등 다른 Tailscale 기기에서 보려면 `BIND=ts scripts/colab-duck view quick` → `http://<Mac 이름>:8080` (tailnet 안에서만 열림).
- VS Code 창 자체를 GPU 머신에 붙이고 싶으면 `Remote-SSH: Connect to Host...` → `colab-duck`. 포트 8080·6006이 자동으로 포워딩된다. 접속하면 세션이 없을 때 새로 만들어지니, 다 쓰면 `colab stop -s duck`.
- 노트북(브라우저 Colab·Kaggle이나 VS Code Colab 커널)은 학습과 ONNX 내보내기만 한다. 결과는 HF Hub에 올라가니 Mac에서 `scripts/duck-local pull` 후:
  ```sh
  scripts/duck-local view quick     # Mac CPU로 3D 뷰어 → http://localhost:8090
  scripts/duck-local tb             # TensorBoard → http://localhost:6007
  ```
- Claude용 규칙은 `CLAUDE.md`, CLI 사용법 스킬은 `.claude/skills/colab-operator`.

## 업데이트

```sh
git -C microduck_rl pull && (cd microduck_rl && uv sync)
git -C microduck-lab pull          # lab 안의 고정 sha 클론은 AGENTS.md의 sha가 바뀌면 그에 맞춰 checkout
```
