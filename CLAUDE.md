# microduck workspace: Claude 작업 메모

폴더 구성과 로컬 실행법은 README.md. 여기에는 Claude가 Colab GPU와 VS Code로 작업할 때 지킬 것만 적는다.

## Colab GPU 작업: `scripts/colab-duck`

- GPU가 필요한 일(학습, 정책 재생)은 `scripts/colab-duck`으로 한다. google-colab-cli(`colab`)와 ssh를 감싼 스크립트이며 명령 목록은 `scripts/colab-duck help`.
- 기준은 로컬 `microduck_rl/`이다. VM의 `/content/microduck_rl`은 복사본이므로, 로컬에서 코드나 보상을 고친 뒤 `scripts/colab-duck sync`로 올린다. VM에서 직접 고친 것은 사라진다.
- **돈이 드는 명령**: `up`(런타임 생성), `train`, 그리고 런타임을 켜 둔 채로 두는 것. `up`과 `train`은 사용자가 요청했을 때만 실행한다. 작업이 끝났다고 하면 `pull` 또는 `export`로 결과를 받은 다음 `down`을 제안한다.
- 학습은 VM 백그라운드에서 돈다(`setsid nohup`). 진행 상황은 `logs NAME`으로 가끔 확인하고, 끝날 때까지 터미널을 붙잡고 있지 않는다.
- `view`와 `tb`는 Ctrl-C를 누를 때까지 끝나지 않는 포트 포워딩이다. 사용자가 직접 VS Code 터미널이나 Task로 실행하게 하고, Claude가 실행할 때는 백그라운드로 돌린다.
- Colab VM의 8080 포트는 Colab 자체 서버(node)가 쓴다. 그래서 viser 뷰어는 8081 같은 다음 빈 포트로 간다. 포트는 `viewer.log`의 `viser (listening *:PORT)`에서 읽는다(스크립트와 노트북 모두 그렇게 함).
- VM의 백그라운드 작업은 `/content/{train,view,tb}.pid`로 관리한다. `pgrep -f`나 `pkill -f`로 이름을 찾으면 ssh로 넘긴 명령줄 자체가 걸리니 쓰지 않는다.
- Colab은 런타임당 SSH 동시 접속 수가 제한돼 있다. 스크립트는 연결 하나(`~/.ssh/cm-colab-duck`)를 나눠 쓰므로 `colab ssh -- CMD`를 따로 병렬로 실행하지 않는다. 429 오류가 나면 VS Code Remote-SSH 같은 다른 접속이 있는지 확인한다.
- VM은 언제든 회수될 수 있다. 체크포인트는 50회마다 VM에 저장되므로 오래 돌린 결과는 중간중간 `pull`로 받는다. 끊기면 `up` → `train NAME ... --resume` 순서로 이어 간다. `--resume`은 VM에 체크포인트가 남아 있어야 쓸 수 있다.
- CLI를 직접 다뤄야 하면 `.claude/skills/colab-operator` 스킬을 본다. `colab auth`, `colab drivemount`, 대화형 `colab ssh`/`repl`은 사람이 터미널에서 해야 한다.
- CLI 로그인(OAuth)도 사람이 해야 한다. 터미널에서 `colab sessions`를 실행하고, 출력된 URL을 승인한 뒤 코드를 붙여 넣는다. 토큰은 macOS 키체인에 저장된다. 키체인이 잠긴 곳(예: ssh로 접속한 셸)에서 로그인하면 `-25308` 경고와 함께 저장이 안 되니, 키체인이 열린 곳에서 다시 로그인한다. 사람이 Remote Control로 접속해 있을 때는, Claude가 named pipe로 `colab sessions`를 띄워 URL을 넘기고, 사람이 붙여 준 코드를 파이프에 써 넣는 방식이 통했다.

## 노트북 (`notebooks/microduck_rl_colab.ipynb`)

- 입문자용 튜토리얼이다(README의 Colab·Kaggle 배지). 브라우저 Colab과 Kaggle 양쪽에서 돌아야 한다. 첫 설치 셀이 `PLATFORM`(colab/kaggle)을 판단한다.
- 사용자가 Colab·Kaggle을 직접 띄우고, Claude는 돕는 역할이다(2026-11-19 결정). 사용자가 요청하지 않으면 `colab-duck up`/`train`을 쓰지 않는다.
- 결과 저장소는 Hugging Face Hub 비공개 저장소 `<아이디>/microduck-runs`(Mac 계정은 `fredisbusy`)다. Drive는 Kaggle에서 못 쓰므로 쓰지 않는다. 토큰은 노트북 비밀값 `HF_TOKEN`이고, 학습 중 60초마다 `logs/rsl_rl`의 체크포인트·params·tfevents·`train_logs/`를 올린다. ONNX는 `exports/`에 올린다. `latest_checkpoint()`는 로컬에 없으면 HF에서 받아 오므로 플랫폼을 바꿔 이어 학습할 수 있다.
- 노트북은 학습과 ONNX 내보내기만 한다. 3D 뷰어와 TensorBoard는 노트북에 띄우지 않는다(사용자 결정). Mac에서 `scripts/duck-local pull`(HF → `runs/hf`) 후 `view RUN`(:8090), `tb`(:6007)로 본다. 학습 로그를 봐 달라고 하면 pull 한 뒤 `runs/hf/train_logs/RUN.log`를 읽는다.
- NotebookEdit을 쓰기 전에 사용자가 노트북을 저장했는지 확인한다. 워크스페이스 설정이 자동 저장이지만, 편집 직전의 변경은 반영 안 됐을 수 있다. NotebookEdit은 고친 셀의 출력을 지운다.
- `mcp__ide__executeCode`는 VS Code 안의 Claude Code에서만 쓸 수 있다. 코드를 노트북 맨 끝에 새 셀로 넣고, 실행할 때마다 사용자 확인이 필요하다.
- 노트북은 VS Code가 저장한 형식이다(`source`는 줄 목록, 키 정렬, indent 1, 파일 끝 줄바꿈). 스크립트로 고칠 때 `json.dumps(nb, ensure_ascii=False, indent=1, sort_keys=True) + "\n"`으로 쓴다.
