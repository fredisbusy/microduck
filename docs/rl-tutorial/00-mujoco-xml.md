# 0단계: MuJoCo XML로 몸 만들기

[← 목차](README.md) · [다음: 1단계 →](01-mujoco-python.md)

## 목표

- XML 파일 하나로 수레와 막대를 만든다.
- MuJoCo 뷰어로 띄워서 손으로 밀어 본다.
- 몸(body), 관절(joint), 모양(geom), 모터(actuator)의 관계를 이해한다.

## 알아 둘 개념

MuJoCo는 로봇을 **MJCF**라는 XML 형식으로 정의한다. 오리 로봇도 같은 형식이다. 핵심 태그는 네 개다.

| 태그 | 뜻 | CartPole에서 |
|---|---|---|
| `<body>` | 움직이는 덩어리. 안에 다른 body를 넣으면 부모-자식으로 붙는다 | 수레(`cart`), 그 안에 막대(`pole`) |
| `<joint>` | body가 부모에 대해 움직일 수 있는 방식 | 수레는 좌우로 미끄러짐(`slide`), 막대는 회전(`hinge`) |
| `<geom>` | 눈에 보이고 부딪히는 모양. 질량도 여기서 준다 | 수레는 상자(`box`), 막대는 캡슐(`capsule`) |
| `<actuator>` | 관절을 움직이는 모터 | 수레를 미는 모터 하나(`push`) |

관절이 없는 body는 부모에 붙박이다. `worldbody` 바로 아래의 geom(바닥, 레일)은 세상에 고정돼 있다.

## 실습

### 1) `cartpole.xml` 만들기

`rl-tutorial/cartpole.xml`을 만들고 아래를 붙여 넣는다. `TODO` 세 곳을 채운다.

```xml
<mujoco model="cartpole">
  <!-- 물리 계산 간격: 0.01초 -->
  <option timestep="0.01"/>

  <worldbody>
    <light pos="0 0 6"/>
    <geom name="floor" type="plane" size="4 4 0.1" pos="0 0 -0.05" rgba=".2 .3 .4 1"/>
    <!-- 레일: 보기용. contype/conaffinity=0 으로 충돌을 끈다 -->
    <geom name="rail" type="capsule" fromto="-2 0 1  2 0 1" size="0.02" rgba=".3 .3 .3 1"
          contype="0" conaffinity="0"/>

    <body name="cart" pos="0 0 1">
      <!-- TODO 1: x축(1 0 0)을 따라 미끄러지는 관절. 이름 slider, 범위 -1.8~1.8 -->
      <joint name="slider" type="____" axis="_ _ _" range="-1.8 1.8" limited="true" damping="5e-4"/>
      <geom name="cart" type="box" size="0.2 0.15 0.1" mass="1" rgba=".7 .1 .1 1"/>

      <body name="pole">
        <!-- TODO 2: y축(0 1 0)을 축으로 도는 관절. 이름 hinge -->
        <joint name="hinge" type="____" axis="_ _ _" damping="2e-6"/>
        <!-- 막대: 수레 중심(0 0 0)에서 위로 1m -->
        <geom name="pole" type="capsule" fromto="0 0 0  0 0 1" size="0.045" mass="0.1" rgba=".1 .5 .8 1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <!-- TODO 3: slider 관절을 미는 모터. 입력 -1~1에 gear 10을 곱해 최대 10N -->
    <motor name="push" joint="______" gear="10" ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>
```

### 2) 뷰어로 띄우기

```sh
uv run python -m mujoco.viewer --mjcf=cartpole.xml
```

창이 뜨면 막대가 서 있다가 곧 한쪽으로 쓰러진다. 정상이다. 아무도 균형을 잡아 주지 않기 때문이다.

### 3) 손으로 만져 보기

| 조작 | 방법 |
|---|---|
| 시뮬레이션 처음으로 | `Backspace` |
| 일시 정지 / 재개 | `Space` |
| 물체 선택 | 더블클릭 |
| 선택한 물체를 끌어당기기 | `Ctrl` + 오른쪽 버튼 드래그 |
| 모터 직접 움직이기 | 오른쪽 패널 **Control** → `push` 슬라이더 |
| 관절 값 보기 | 오른쪽 패널 **Joint** |

`Backspace`로 처음으로 돌린 뒤 `push` 슬라이더를 좌우로 움직여 막대를 세워 보자. 사람도 꽤 어렵다. 이걸 2단계에서 신경망이 배운다.

## 확인

- [ ] 수레가 좌우로만 움직이고, 레일 끝(±1.8m) 근처에서 멈춘다.
- [ ] 막대가 수레에 붙은 채 앞뒤로 쓰러진다(옆으로는 안 쓰러진다).
- [ ] `push` 슬라이더로 수레를 움직일 수 있다.
- [ ] **Joint** 패널에 `slider`와 `hinge` 두 줄이 보인다.

## 더 해 보기

한 번에 하나만 바꾸고 뷰어를 다시 띄워 차이를 본다(`Backspace`로 다시 시작).

1. 막대 길이 `fromto="0 0 0  0 0 1"`의 마지막 `1`을 `0.5`로 줄이면? 쓰러지는 속도가 어떻게 달라지는가?
2. 막대 질량 `mass="0.1"`을 `1`로 올리면? `push`로 세우기가 쉬워지는가, 어려워지는가?
3. 모터 `gear="10"`을 `2`로 줄이면? 힘이 약하면 막대를 살릴 수 있는가?
4. 레일의 `contype="0" conaffinity="0"`을 지우면? 수레가 움직이는가?

4번이 중요하다. 레일 캡슐이 수레 상자를 관통하고 있어서, 충돌을 켜면 수레가 레일에 끼어 꼼짝 못 한다. 이 튜토리얼을 만들 때 실제로 이 실수 때문에 2단계 학습이 전혀 안 됐다. 학습이 안 될 때 가장 먼저 의심할 곳은 알고리즘이 아니라 **환경**이다.

<details>
<summary>정답</summary>

```xml
<joint name="slider" type="slide" axis="1 0 0" range="-1.8 1.8" limited="true" damping="5e-4"/>
...
<joint name="hinge" type="hinge" axis="0 1 0" damping="2e-6"/>
...
<motor name="push" joint="slider" gear="10" ctrlrange="-1 1" ctrllimited="true"/>
```

- `slide`는 한 축으로 미끄러지는 관절, `hinge`는 한 축으로 도는 관절이다.
- 수레가 x축으로 움직이니 막대는 x-z 평면에서 쓰러져야 한다. 그 평면에 수직인 y축(`0 1 0`)이 회전축이다.

더 해 보기 답:
1. 짧은 막대가 더 빨리 쓰러진다. 그래서 짧은 막대가 균형 잡기에 더 어렵다(손바닥 위 연필 vs 빗자루).
2. 막대가 무거우면 같은 힘으로 수레를 덜 움직이는 셈이라 반응이 둔해진다.
3. 힘이 너무 약하면 이미 기운 막대를 되돌리지 못한다. 학습 문제가 "불가능한 문제"가 될 수 있다.
4. 수레가 레일에 끼어 거의 움직이지 않는다.

</details>

## 오리와 연결

오리 로봇도 똑같이 생겼다. 몸통 `body` 안에 다리 `body`들이 `hinge` 관절로 이어져 있고, 관절마다 모터가 붙어 있다. 차이는 개수다. CartPole은 관절 2개·모터 1개이고, 오리는 다리마다 관절과 모터가 여러 개다. 또 걷는 로봇은 보통 `motor`(힘을 직접 줌) 대신 목표 각도를 따라가는 위치 제어 모터를 쓴다. 4단계 뒤에 오리 XML을 열어 직접 확인해 보자.

[다음: 1단계 →](01-mujoco-python.md)
