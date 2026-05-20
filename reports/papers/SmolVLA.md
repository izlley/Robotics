# SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics

> **출처**: Shukor, Aubakirova, Capuano et al. (Hugging Face + Sorbonne + valeo.ai + ENS Paris-Saclay), 2025. arXiv:2506.01844v1 (2 Jun 2025).
> **읽은 일자**: 2026-05-20
> **PDF**: [`papers/core-models/SmolVLA-A-Vision-Language-Action-Model-for-Affordable-and-Efficient-Robotics.pdf`](../../papers/core-models/SmolVLA-A-Vision-Language-Action-Model-for-Affordable-and-Efficient-Robotics.pdf)
> **분량**: 본문 14 페이지 + 부록 10 페이지 = 24 페이지

---

## 한 줄 요약

**450M 경량 VLA**. RT-2/OpenVLA의 "discrete action token" 패러다임 대신 **VLM(SmolVLM-2) + Flow Matching action expert** 구조 채택(= π0 계열과 같은 paradigm). 핵심 효율화 4가지(**layer skipping, visual token 축소, interleaved CA/SA, 비동기 추론**)로 **OpenVLA(7B)·π0(3.3B)와 동등하거나 능가**, training은 6x 적은 메모리 + 40% 빠르게, inference는 consumer GPU·CPU에서 가능.

## TL;DR

- **새 패러다임 첫 등장**: RT-2/OpenVLA는 "action을 token으로" (discrete bin), SmolVLA는 **"flow matching으로 continuous action 생성"** (π0과 동일 흐름). 본 프로젝트 8편에서 처음 만나는 action expert 모델.
- **3가지 핵심 효율화**:
  1. **VLM layer skipping**: SmolVLM-2의 16층 LLM 중 **앞 8층만 사용** → 계산 ½
  2. **Visual token 64개**: tiling 제거, pixel shuffle로 token 수 압축
  3. **Interleaved Cross-Attention + Self-Attention**: action expert 안에 CA/SA를 번갈아 배치 (모든 layer가 둘 다 갖는 대신)
- **VLM은 frozen, action expert만 학습** — RT-2/OpenVLA는 vision까지 다 fine-tune했던 것과 정반대
- **Community datasets (481개, 23K episodes, 10.6M frames)** — OpenVLA 970K 대비 ~42x 적음. SO-100/101 같은 저렴한(<$200) 3D-printed arm으로 수집된 데이터
- **Asynchronous inference**: action 실행과 prediction을 decouple하여 latency 30% 감소, 같은 시간에 2x 작업 처리
- **결과**: LIBERO 87.3% (OpenVLA 76.5%, π0 3.3B의 86.0% 능가/비등). Real SO100에서 π0(3.5B)의 61.7%를 78.3%로 능가
- 한계: single embodiment 학습, 23K episode는 작은 편, OCR 위주로 사전학습된 VLM 사용 (robotics 특화 X)

---

## 1. Motivation & 문제 정의

### 1.1 풀려는 문제

OpenVLA가 "오픈 VLA"라는 첫 장벽을 깼지만, 두 번째 장벽은 **여전히 비싸다**:
- 7B 모델 학습: 64 A100 × 14일 = 21,500 A100-hours
- LoRA fine-tune이라도 single A100 80GB 필요
- 추론: 16GB+ VRAM (RTX 4090급)
- 데이터: 970K episode를 모으려면 산업적 수집 capacity 필요

**커뮤니티(개인 연구자, 학교, 스타트업)가 실제로 진입하려면 한 자릿수 더 작아야 한다**. 그리고 그 작은 모델은 **저렴한 로봇 하드웨어(SO-100 등 ~$200 3D-printed arm)**에서 돌아가야 한다.

### 1.2 기존 방법의 한계

| 접근 | 한계 |
|---|---|
| RT-2, OpenVLA, π0 (>3B) | 학습·추론 모두 무거움. Consumer GPU 불가 |
| TinyVLA (~1B) | Robotics 사전학습 없음 → generalization 약함 |
| Diffusion Policy (~수십 M) | VLM 자산 활용 못 함 → semantic understanding 약함 |
| Octo (93M) | Action policy 작지만 VLM 없음, 결국 성능 ceiling |

→ **VLM 자산 + 경량 + 저렴 robot + 커뮤니티 data + 공개 stack**을 다 만족하는 모델 부재.

### 1.3 본 논문의 가설

> "VLM은 경량(450M급)으로, action 부분은 별도 flow matching expert로 분리하면, 7B급 모델과 동등 성능을 consumer GPU에서 낼 수 있다."

이 가설이 맞다면:
- 학교·개인이 VLA 학습 가능 (single GPU)
- 저렴한 SO-100/101 arm으로 실 robot 실습 가능
- VLA 커뮤니티 진입 장벽 산업체 → 누구나

## 2. 핵심 아이디어

### 2.1 한 줄

**"VLM은 그대로 가볍게 (450M) + action 생성은 별도의 flow matching transformer에 위임"**. 두 모듈은 attention으로 연결.

### 2.2 이게 무엇이 새로운가 — 두 가지 패러다임의 분기점

지금까지 본 RT-2/OpenVLA는 **단일 VLM이 모든 걸 한다**:

```
Image + Text → VLM → action_token (256 bin discrete)
```

SmolVLA는 **VLM과 action expert가 분리**된 구조:

```
Image + Text → VLM (perception only)
                ↓ features
              Action Expert (flow matching) → continuous action chunk
```

이는 π0(2024), GR00T N1, Gemini Robotics와 같은 흐름. **본 프로젝트에서 처음 등장하는 패러다임**.

**왜 분리하는가?**
1. VLM에 action 생성을 시키면 fine-grained continuous control 정밀도가 부족 (256 bin ceiling)
2. Flow matching/Diffusion 같은 generative model은 action distribution을 더 잘 모델링
3. VLM 부분만 frozen + action expert만 학습 → 효율적 (이게 SmolVLA의 핵심 선택)

### 2.3 SmolVLA의 4가지 독특한 효율화

| 효율화 | 메커니즘 | 효과 |
|---|---|---|
| **Layer skipping** | SmolVLM-2의 16층 중 앞 8층(N=L/2)만 사용 | LLM 계산 ½ |
| **Visual token 축소** | Tiling 제거 + pixel shuffle → 프레임당 64 token | 시퀀스 ↓ |
| **Interleaved CA+SA** | 모든 block에 CA+SA 대신 한 block당 하나씩 번갈아 | Params ↓, 동등 성능 |
| **Async inference** | Action 실행과 prediction을 별 process로 분리 | Latency 30%↓ |

### 2.4 LLM/VLM 도구와의 analogy

| LLM 개념 | SmolVLA 대응 |
|---|---|
| Frozen LLM + adapter (LoRA, prefix) | Frozen VLM + Action Expert (separate small transformer) |
| Speculative decoding | Async inference (다른 메커니즘이지만 같은 목표) |
| Layer pruning (DistilBERT) | First-N-layers 사용 |
| Mixture-of-Experts | (없음, but action expert는 single expert로 봐도 됨) |
| Flow matching for diffusion | Flow matching for action |
| LM tokens | Visual tokens reduction + action chunk tokens |

## 3. 아키텍처 (상세)

### 3.1 입력 / 출력

| 항목 | 형식 | 차원 | 비고 |
|---|---|---|---|
| Vision | 다중 RGB 카메라 (top, wrist, side 표준화) | 각 512×512 → 64 token per view | OpenVLA(1 image)와 다름 |
| Language | 자연어 instruction | tokenized | Llama-style tokenizer |
| Sensorimotor state | Joint angles 등 robot 상태 | **1 token** (linear projection) | OpenVLA는 입력 없음 |
| Output | Continuous action chunk | n=50 actions × action_dim | π0과 같은 chunking |

**핵심 차이 from OpenVLA**:
1. **State 입력 명시적**: joint state를 linear projection하여 token으로 추가 (prefix attention)
2. **Multi-view**: top, wrist, side 카메라 (저자가 community data를 표준화)
3. **Continuous action**: discrete bin 없음

### 3.2 전체 구조 다이어그램

```
┌──────────────────────────────────────────────────────────────┐
│ Inputs                                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │Top cam      │  │Wrist cam    │  │ "Grasp the object   │  │
│  │RGB 512x512  │  │RGB 512x512  │  │  and put it in bin" │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
└─────────┼─────────────────┼─────────────────────┼────────────┘
          ▼                 ▼                     ▼
     ┌─────────────────────────┐         ┌───────────────┐
     │  SigLIP vision encoder  │         │  Tokenizer    │
     │  (per-view, pixel-      │         └───────┬───────┘
     │   shuffle → 64 tok/view)│                 │
     └────────────┬────────────┘                 │
                  │                              │
   robot state ──→ linear proj ──→ 1 state token │
                  │                              │
                  ▼   ▼      ▼                   ▼
              ┌──────────────────────────────────┐
              │ SmolLM2 (LLM in SmolVLM-2)       │
              │ ⚠ FIRST N=L/2=8 LAYERS ONLY ⚠   │
              │ (16층 중 앞 8층만 사용)            │
              │ → features at layer N (frozen)   │
              └──────────────┬───────────────────┘
                             │ VLM features
                             ▼
              ┌──────────────────────────────────┐
              │ Action Expert v_θ                │
              │ (FLOW MATCHING transformer)      │
              │ ┌──────────────────────────┐    │
              │ │ Block 1: Cross-Attention │ ←── KV from VLM features
              │ │ Block 2: Self-Attention  │ ←── causal mask, on action tokens
              │ │ Block 3: CA              │    │
              │ │ Block 4: SA              │    │
              │ │   ...                    │    │
              │ │ (interleaved)            │    │
              │ └──────────────────────────┘    │
              │ Width: 0.5d (절반)               │
              └──────────────┬───────────────────┘
                             │
                             ▼
                  Action chunk: n=50 actions
                  A_t, A_{t+1}, ..., A_{t+49}
                             │
                             ▼
                  Robot controller (async, ~30% faster)
```

### 3.3 핵심 모듈 1: VLM Backbone — SmolVLM-2

**SmolVLM-2** (Marafioti et al. 2025)는 Hugging Face의 경량 VLM. 구성:
- **Vision encoder**: SigLIP (∼300M)
- **Language decoder**: **SmolLM-2** (∼135M, 16 transformer layers)
- 합치면 ~450M params

**SmolVLA가 가져다 쓰는 방법** (3가지 수정):

**(a) Layer skipping (가장 중요)**

SmolVLM-2의 SmolLM-2 LLM은 16층. SmolVLA는 **앞 8층만 사용**하고 나머지 8층은 **버린다**:

$$
\text{features} = \text{SmolLM-2}_{1:N}(input), \quad N = L/2 = 8
$$

**왜 이게 작동하나?**
- 최근 연구 (El-Nouby 2024, Bolya 2025, Rajasegaran 2025): VLM의 best downstream feature는 마지막 layer가 아닐 수 있다
- Robot 제어에 필요한 spatial/semantic feature가 중간층(layer 8 정도)에서 이미 충분
- 마지막 8층은 자연어 generation에 특화 → robot action에는 불필요

**Ablation 결과** (Table 8, LIBERO Avg):

| 사용 layer | LIBERO Avg | 의미 |
|---|---|---|
| 8 (=L/2) | 75.0 | 절반만 |
| 16 (=L) | 78.5 | 전체 |
| 24 | 79.5 | 더 큰 VLM (의미 없음, SmolVLM-2가 16층) |
| 32 | 80.3 | 더더 큰 VLM |
| **VLM-256M (전체)** | **75.8** | **작은 VLM 전체** ← 비교 |
| Skip %2 (16층 중 격수 8층) | 75.5 | 균등 skip |

해석:
- **첫 8층 사용 = VLM-256M (작은 VLM) 전체 사용** 거의 동일 성능
- 즉 **큰 VLM의 절반 = 작은 VLM 통째**보다 약간 좋음
- "큰 모델을 만들고 잘라쓰는 게 작은 모델 통째 쓰는 것보다 같거나 좋다" — pruning research의 robotics 응용

**(b) Visual tokens 축소**

SmolVLM-2 default는 **image tiling** (multiple crops + global). SmolVLA는:
- Tiling 제거 (단일 global image만)
- **Pixel shuffle** (Lin 2023b)로 patch token을 ½ × ½ = ¼ 압축
- 결과: **frame당 64 visual tokens** (보통 VLM은 ~256+)

이는 추론 속도에 결정적 (transformer cost는 sequence length 제곱에 비례).

**(c) State token**

Robot state $s_t$ (joint angles 등)는 단순 linear projection으로 1 token으로 변환:
$$
\text{state\_tok} = W_s \cdot s_t + b_s, \quad W_s \in \mathbb{R}^{d_s \times d_{\text{vlm}}}
$$
이 1 token이 visual+language tokens과 concat되어 VLM에 들어감 (prefix).

### 3.4 핵심 모듈 2: Flow Matching Action Expert

**이 부분이 RT-2/OpenVLA와 결정적으로 다른 곳**. 처음 만나는 개념이라 자세히.

#### 3.4.1 Flow Matching 개념 (한 페이지로)

**Diffusion model** 알면 거의 같다. 차이는:
- **Diffusion**: 시간을 따라 noise를 점진적으로 추가/제거. score function 학습.
- **Flow Matching** (Lipman 2022, Liu 2022): noise → data로 가는 **직선 path**의 **velocity field**를 학습. ODE 한 번 풀면 끝.

수학적으로:

- Ground truth action chunk: $A_t = (a_t, a_{t+1}, ..., a_{t+n-1})$
- Noise: $\epsilon \sim \mathcal{N}(0, I)$
- Interpolated noisy action ($\tau \in [0, 1]$):
$$A_t^\tau = \tau A_t + (1 - \tau) \epsilon$$

- 학습: velocity field $v_\theta(A_t^\tau, o_t)$를 학습해서 ground-truth velocity $u(A_t^\tau | A_t)$에 맞추기:
$$u(A_t^\tau | A_t) = \frac{d A_t^\tau}{d \tau} = A_t - \epsilon$$

논문 표기에 따라 부호가 $\epsilon - A_t$인데, 결과적으로 같은 정보 (방향만 반대):
$$\boxed{\mathcal{L}_\tau(\theta) = \mathbb{E}_{p(A_t | o_t), q(A_t^\tau | A_t)} \left\| v_\theta(A_t^\tau, o_t) - u(A_t^\tau | A_t) \right\|^2}$$

기호:
- $o_t$: VLM features (perception output)
- $v_\theta$: action expert (transformer)
- $\tau$: flow parameter, **Beta 분포에서 sampling** (π0 따라함)
- 손실: 단순 MSE

**Inference 시**: $\tau = 0$ (pure noise)에서 시작해 ODE를 따라 $\tau = 1$ (clean action)까지 적분. SmolVLA는 **10 step**으로 적분 (충분히 정확).

```python
# pseudocode (inference)
def sample_action(o_t):
    A = noise()  # sample from N(0, I)
    for tau in linspace(0, 1, 10):  # 10 steps
        v = v_theta(A, o_t, tau)
        A = A + v * (1/10)        # Euler integration
    return A  # clean action chunk
```

**왜 token-based 대신 flow matching?**
- Token: discrete 256 bin, 정밀도 한계, 단일 예측만
- Flow matching: continuous, multimodal action distribution 표현 가능 (한 상황에 여러 valid action), smoother trajectory

#### 3.4.2 Interleaved CA + SA (논문의 새로움)

기존 transformer block: **Self-Attention + Cross-Attention 둘 다** 한 block에 있음. SmolVLA의 action expert는:

```
Block 1: Cross-Attention (KV from VLM features)
Block 2: Self-Attention (causal, on action tokens)
Block 3: Cross-Attention
Block 4: Self-Attention
...
```

**왜 분리?**
- **Params 절감**: 한 block이 둘 다 가지면 params 2x. 분리하면 ½.
- **Inference 빠름**: layer가 좁아져서 FLOPs ↓
- **성능은 동등**: 정성적으로 SA가 action chunk를 부드럽게 만들고, CA가 VLM 신호를 받아옴 → 둘이 보완

**Ablation 결과** (Table 6, LIBERO Avg):

| Attention 구성 | LIBERO Avg |
|---|---|
| CA only | 79.0 |
| SA only | 74.5 |
| **CA + SA interleaved** | **85.5** ★ |

→ 둘 다 필요. Interleaved가 best.

**SA에서 causal mask 사용** (Table 7):

| SA mask | LIBERO Avg |
|---|---|
| Bidirectional | 67.5 |
| **Causal** | **74.5** |

action chunk 안에서 미래 action에 attend하지 못하게 → leakage 방지 → 성능 ↑.

#### 3.4.3 Action Expert 폭(width)

VLM hidden dim $d$ 대비:

| Expert width | LIBERO Avg |
|---|---|
| 1.00 d | 82.3 |
| **0.75 d** ← 논문 본문 default | **77.5** |
| 0.50 d ← ablation에서 사용 | 80.3 |
| 0.25 d | 73.8 |

→ 0.5~0.75d 사이가 sweet spot. 0.25d로 너무 줄이면 capacity 부족.

### 3.5 핵심 모듈 3: Asynchronous Inference

**문제**: 보통 robot policy 추론은 sync 방식:
1. Observation $o_t$ 캡처
2. Action chunk $A_t = (a_t, ..., a_{t+n-1})$ 예측 (시간 $\ell$ 소요)
3. Chunk 전체 실행
4. 다음 chunk를 위해 다시 (1)로

**문제점**: (2)에서 inference 동안 robot은 **idle**. Round-trip latency $\ell$ (예: 500ms)이 그대로 손실.

**Async inference**: action 실행과 prediction을 **두 process**로 분리.

```
RobotClient (loop):                  PolicyServer:
  while not done:                       (always listening)
    a = popfront(action_queue)          on observation o:
    execute(a)                            A = predict(o)
    if len(queue) < g·n:                  send back A
      o_new = capture()
      if not duplicate(o_new):
        async_call(server, o_new)
    if server_response:
      queue = merge(queue, A_new)
```

**Threshold $g \in [0, 1]$**: 현재 queue의 길이가 $g \cdot n$ 아래로 떨어지면 새 inference 요청. 큐가 비기 전에 새 chunk가 도착하도록 (idle 방지).

**수학적 분석**:
- $E[\ell_S]$: server inference time
- $\Delta t$: control cycle (예: 30 fps에서 33ms)
- **Idle 방지 조건**: $g \geq E[\ell_S] / (\Delta t \cdot n)$
- Sweet spot: $g \approx 0.7$ (논문 추천)

**3가지 한계 시나리오**:
- $g = 0$: 완전 sync (idle 발생)
- $g = 0.7$: balanced (논문 추천)
- $g = 1.0$: 매 timestep마다 inference (compute 부담, queue 거의 가득)

**Near-duplicate filter**: joint space에서 두 observation이 너무 비슷하면 (distance $< \epsilon$) inference 안 함. 불필요한 server call 방지.

**결과** (Figure 5, Pick-Place task):
- Sync: 평균 13.75s/task, 60s 동안 9 cubes
- **Async: 평균 9.7s/task, 60s 동안 19 cubes** (2x 처리량!)

이 async 메커니즘은 **model-agnostic** — action chunk를 출력하는 모든 policy에 적용 가능.

## 4. 데이터 (상세)

### 4.1 Community Datasets (가장 큰 contribution)

기존 VLA 학습 데이터:
- RT-2: ~130K episode (RT-1 dataset, single embodiment)
- OpenVLA: 970K episode (OpenX, 22+ embodiments, 산업적 수집)
- π0: ~10,000 hours cross-embodiment data

**SmolVLA**: **HuggingFace에 사용자들이 올린 community datasets 481개**, 총 **22.9K episodes, 10.6M frames**.

| 항목 | 수치 | 비교 |
|---|---|---|
| Datasets | 481 | LeRobot 표준 포맷 |
| Episodes | **22.9K** | OpenVLA 970K의 ~2.4% |
| Frames | **10.6M** | OpenVLA 추정 (1 episode = 50 step → 48.5M frame)의 ~22% |
| Embodiments | **거의 SO-100 단일** | OpenVLA 22+ |
| Quality | **노이즈 많음** (community contribution) | OpenVLA는 cleaned |

**왜 이게 의미 있나?**
- 산업적 자원 없이도 학습 가능
- "**Community data가 robotics의 ImageNet 역할** 할 수 있음을 입증"
- 단점: 노이즈 + 단일 embodiment → 일반화 한계

### 4.2 데이터 표준화 과제

Community data가 모이면 다음 문제들이 생긴다:

**(a) Task annotation noise**

원본 데이터의 instruction이:
- `"task_desc"` (placeholder)
- `"Hold"`, `"Up"` (너무 모호)
- 없는 경우도 있음

**해결**: **Qwen2.5-VL-3B-Instruct VLM으로 자동 재annotation**.

Prompt:
```
Here is a current task description: {current_task}. 
Generate a very short, clear, and complete one-sentence
describing the action performed by the robot arm (max 30 characters).
Be concise. Start directly with an action verb like "Pick", "Place", "Open", etc.
```

→ 모든 task가 "Pick up the red cube" 같은 표준 형식으로 통일.

**LLM 엔지니어 관점**: 이건 **synthetic data refinement** 그대로. LLM으로 데이터 cleaning하는 패턴이 robotics에도 적용됨.

**(b) Camera viewpoint 표준화**

각 데이터셋이 카메라 이름을 임의로 사용:
- `images.laptop`이 top일 수도 wrist일 수도 side일 수도
- → 학습 시 혼선

**해결**: 수동으로 `OBS_IMAGE_1` (top), `OBS_IMAGE_2` (wrist), `OBS_IMAGE_3` (side)로 매핑. 추가 view는 drop.

저자도 "미래에는 VLM으로 자동화 가능"이라 인정.

### 4.3 학습 데이터 vs Robot 데이터

훨씬 더 작은 robot data로도 OpenVLA를 능가:

| 모델 | Episodes | 추론 모델 크기 |
|---|---|---|
| SmolVLA | **23K** | **0.45B** |
| OpenVLA | 970K | 7B |

**Implication**: VLM 자산이 충분히 크면 robot data가 적어도 OK. 데이터 양보다 **VLM 사전학습의 질**과 **architecture 효율화**가 더 중요.

## 5. 학습 (상세)

### 5.1 Loss

위의 flow matching loss (3.4.1):
$$
\mathcal{L}_\tau(\theta) = \mathbb{E}_{(o_t, A_t)} \mathbb{E}_{\tau \sim \text{Beta}, \epsilon \sim \mathcal{N}(0, I)} \left\| v_\theta(\tau A_t + (1-\tau)\epsilon, o_t) - (\epsilon - A_t) \right\|^2
$$

**기호**:
- $\theta$: **action expert만** ($v_\theta$). VLM은 frozen.
- $o_t$: VLM의 layer 8 features (perception)
- $A_t$: ground-truth action chunk (size n=50)
- $\tau$: flow time, **Beta distribution sampling** (π0과 동일. 보통 Beta(1.5, 1)로 $\tau$가 중간값에 집중)
- $\epsilon$: standard Gaussian noise

**중요**: VLM은 **frozen**. 학습되는 건 action expert뿐. OpenVLA(vision encoder까지 unfreeze)와 정반대 결정.

**왜 VLM frozen?**
- SmolVLA의 핵심 목표: 효율. Action expert만 학습 = 학습 비용 ↓
- VLM features가 robot 제어에 충분히 강력하다는 가정 (실제로 ablation에서 확인)
- π0과 같은 전략

### 5.2 Optimization

| Hyperparameter | 값 |
|---|---|
| Steps (pretraining) | **200K** |
| Batch size | 256 |
| LR warmup | 100 steps |
| LR schedule | Cosine (1e-4 → 2.5e-6) |
| Optimizer | AdamW, $\beta_1 = 0.9$, $\beta_2 = 0.95$ |
| Image resolution | 512×512 (SmolVLM-2 expected) |
| Action chunk size $n$ | **50** |
| Flow matching steps (inference) | **10** |
| Pretraining GPU hours | **30K total** (4 GPUs 사용했지만 single GPU도 가능) |
| Precision | bfloat16 + torch.compile() |

**Fine-tuning** (downstream tasks):
- Simulation (LIBERO, Meta-World): 100K steps, batch 64
- Real-world (SO-100/101): 200K steps
- 저자는 "훨씬 적은 step으로도 OK"라 언급

### 5.3 학습 비용 비교

| 모델 | 사전학습 비용 |
|---|---|
| OpenVLA | 21,500 A100-hours |
| **SmolVLA** | **~30K GPU-hours** (대부분 V100 또는 A100) |
| π0 | (논문 미공개, but >10K hours 추정) |

논문 주장: **single GPU로 학습 가능**. 4 GPU를 쓴 건 batch 256을 위해. 만약 batch 64로 줄이면 single GPU로도 됨.

### 5.4 모델 사이즈 라인업

논문은 3가지 size 평가:

| Size | Params | LIBERO Avg | Meta-World Avg |
|---|---|---|---|
| SmolVLA 0.24B | 240M | 82.75 | 56.95 |
| **SmolVLA 0.45B** ← 본문 default | **450M** | **87.3** | 57.3 |
| SmolVLA 2.25B | 2.25B | 88.75 | 68.24 |

→ 0.45B가 sweet spot. 2.25B는 더 좋지만 efficiency 목표와 충돌.

## 6. 평가 (상세)

### 6.1 Setup

**Simulation**:
- **LIBERO** (Liu 2023a): 4 카테고리(Spatial, Object, Goal, Long) × 10 task = 40 task
- **Meta-World** (Yu 2020): 50 task, difficulty(Easy, Medium, Hard, Very Hard)

**Real-world**:
- **SO-100** (저렴 3D-printed arm): Pick-place, Stacking, Sorting (multi-task)
- **SO-101** (SO-100의 개량형): Pick-Place-Lego (single-task)

**Robots**:
- SO-100, SO-101: 6-DoF, ~$200 3D-printed 로봇 (LeRobot 표준)
- Panda (LIBERO sim): Franka Emika 7-DoF
- Sawyer (Meta-World sim): 4-DoF

### 6.2 Baselines

| Baseline | 크기 | 특징 |
|---|---|---|
| ACT (Zhao 2023) | 80M | CVAE policy, from-scratch |
| Diffusion Policy (Chi 2023) | ~수십M | Action diffusion, from-scratch |
| Octo (Team 2024) | 93M | Open generalist policy |
| OpenVLA (Kim 2024) | **7B** | Token-based VLA (우리 정독 완료) |
| **π0 PaliGemma-3B** | 3.5B | VLM-init만, robotics pretrain 없음 |
| **π0 (full)** | 3.3B | Robotics pretrain 포함 |
| TinyVLA (Zhou 2024) | <1B | 경량 baseline |

### 6.3 핵심 결과 1 — Simulation (Table 2)

**LIBERO** Avg Success Rate (%):

| Model | Params | VLA Pretrain | Avg |
|---|---|---|---|
| Diffusion Policy | ~M | No | 72.4 |
| Octo | 0.09B | Yes | 75.1 |
| **OpenVLA** | **7B** | Yes | **76.5** |
| π0 (PaLiGemma-3B) | 3B | No | 71.8 |
| π0 (full robotics PT) | 3.3B | **Yes** | **86.0** |
| **SmolVLA 0.45B** | 0.45B | **No** | **87.3** ★ |
| SmolVLA 2.25B | 2.25B | No | 88.75 |

해석:
- SmolVLA 0.45B(robotics pretrain 없음)가 **OpenVLA 7B를 +10.8pp** 능가
- π0 3.3B(robotics pretrain 있음)와 비등 (87.3 vs 86.0)
- π0(3B)이 robotics pretrain 없을 때 71.8%인데, SmolVLA가 같은 조건에서 87.3% → **architecture가 결정적**

**Meta-World** Avg Success Rate (%):

| Model | Avg | 비고 |
|---|---|---|
| Diffusion Policy | 10.5 | hard task에서 거의 fail |
| TinyVLA | 31.6 | |
| π0 (PaLiGemma-3B) | 50.5 | |
| π0 (full) | 47.9 | |
| **SmolVLA 0.45B** | **57.3** | ★ |
| SmolVLA 2.25B | 68.24 | |

→ Hard/Very Hard task에서도 SmolVLA가 강함 (60% Very Hard).

### 6.4 핵심 결과 2 — Real-World SO-100 (Table 3)

| Policy | Pick-Place | Stacking | Sorting | **Avg** |
|---|---|---|---|---|
| ACT (single-task FT) | 70 | 50 | 25 | 48.3 |
| π0 3.5B (multi-task) | 100 | 40 | 45 | 61.7 |
| **SmolVLA 0.45B (multi-task)** | 75 | **90** | **70** | **78.3** ★ |

해석:
- π0 3.5B(7x larger)를 +16.6pp 능가
- 특히 Stacking에서 50pp 차이 (90 vs 40) — 작은 모델이 더 잘함!
- Pick-Place에서만 π0이 100% (SmolVLA 75%) — 가장 simple task에서는 π0 우위

### 6.5 핵심 결과 3 — Cross-embodiment Generalization (Table 4)

SmolVLA는 **SO-100만으로 사전학습**됨. SO-101에 fine-tune:

| Policy | In Distribution | OOD |
|---|---|---|
| ACT (single-task) | 70 | 40 |
| **SmolVLA 0.45B** | **90** | **50** |

→ SO-100 → SO-101 (다른 embodiment)도 fine-tune으로 잘 transfer.

### 6.6 Ablation — Pretraining + Multi-task 효과 (Table 5)

| Setting | Pick-Place | Stacking | Sorting | Avg |
|---|---|---|---|---|
| SmolVLA, single-task FT only | 55 | 45 | 20 | **40** |
| SmolVLA, multi-task FT only | 80 | 40 | 35 | 51.7 |
| **SmolVLA, community pretrain + multi-task FT** | **75** | **90** | **70** | **78.3** ★ |

해석:
- **Pretraining +27pp** (40 → 78.3)
- **Multi-task FT +12pp** vs single-task
- Pretraining이 multi-task FT보다 더 큰 효과

### 6.7 Ablation — Async Inference 효과 (Figure 5)

| Mode | Success | Time/task | 60s 내 작업 수 |
|---|---|---|---|
| Sync | 78.3% | 13.75s | 9 |
| **Async** | 73.3% (slightly lower) | **9.7s** | **19** |

→ Success rate는 약간 떨어지지만 (1 task에서 sorting 50% vs 70% 등 sync가 더 잘) **시간당 처리량 2x**.

### 6.8 Ablation — 핵심 설계 결정

**Cross-attention vs Self-attention** (Table 6):
- CA only: 79.0%
- SA only: 74.5%
- **CA + SA interleaved: 85.5%** ★

**Causal vs Bidirectional SA** (Table 7):
- Bidirectional: 67.5%
- **Causal: 74.5%** (causal이 +7pp)

**Layer skipping** (Table 8):
- N=8 (=L/2): 75.0
- N=16 (full): 78.5
- 그러나 **N=8이 작은 VLM 256M(75.8)과 비등** → "큰 모델의 절반 ≥ 작은 모델 통째"

**Action expert width** (Table 9):
- 1.00d: 82.3
- 0.75d: 77.5
- **0.50d: 80.3** ← efficient sweet spot
- 0.25d: 73.8

**Loss** (Table 10):
- **Flow matching: 80.25%**
- Regression L1: 75.25%
- → Flow matching이 +5pp. Multimodal action distribution 모델링 우위.

**State prefix vs suffix** (Table 11):
- **Prefix (VLM에 state token 추가): 80.3%** (CA mode)
- Suffix (action expert에 state token 추가): 73.3%
- → State는 VLM에 prefix로 줘야 됨 (VLM이 state-conditioned features를 만들 수 있게)

**Chunk size n** (Table 12):
- n=1: 50% (너무 작음)
- n=10: 84.0
- n=50: 80.3
- n=100: 74.5 (너무 큼)
- → 10~50이 sweet spot

**Observation update frequency** (Table 13):
- Update every 1 action: 80.3 (가장 반응적)
- Update every 10: 82.8
- Update every 50 (full chunk): 51.8 (너무 둔감)
- → Frequent update가 좋음 (async inference의 이유)

## 7. 강점 / 한계

### 7.1 강점

| 강점 | 구조적 원인 |
|---|---|
| Consumer GPU 학습·추론 가능 | 450M + frozen VLM + layer skip |
| 큰 모델 동등 성능 | Flow matching action expert + 효율적 attention 패턴 |
| 작은 데이터로 학습 | 23K episode (OpenVLA 970K의 2.4%) — community data로 충분 |
| Async inference 30% 빠름 | Action 실행/예측 decoupling |
| 저렴한 robot에서 작동 | SO-100/101 (~$200) 학습 + 평가 |
| 풀 오픈 | Code + weights + data + 학습 recipe + 로봇 hardware 모두 공개 |
| LeRobot 통합 | HF 생태계에서 즉시 사용 가능 |
| Multi-camera + state 지원 | OpenVLA가 못 한 부분 |

### 7.2 한계 — Mechanism 분석

| 한계 | 구조적 원인 | 후속 모델이 어떻게 해결? |
|---|---|---|
| **단일 embodiment 학습** | Community data가 거의 SO-100 단일 | 다양한 robot의 community data 필요 (X-VLA 등) |
| **23K episode small** | Community contribution scale 한계 | DROID 같은 대형 data, 또는 더 큰 community data |
| **OCR 위주 VLM 사용** | SmolVLM-2가 document/OCR 중심 사전학습 | Robotics-specific VLM 사전학습 |
| **Imitation only** | RL 없음 | π★0.6 (RECAP), ConRFT |
| **Short-horizon만 잘함** | Action chunk n=50 → 5초 정도 | Hierarchical / world model (π0.7) |
| **No tactile/force** | Vision + language + state만 | Rho-alpha 등 multimodal sensor |

## 8. 다른 모델과의 관계

### 8.1 직접적 선행

- **[[π0]]** (Black 2024): SmolVLA의 직접적 영감 원천. VLM + flow matching action expert 패러다임을 SmolVLA가 그대로 차용. π0은 PaLiGemma-3B 기반, SmolVLA는 SmolVLM-2-450M 기반.
- **[[OpenVLA]]** (Kim 2024): 다른 패러다임이지만 "오픈 VLA" 정신을 SmolVLA가 계승. SmolVLA의 비교 baseline.
- **SmolVLM-2** (Marafioti 2025): SmolVLA의 VLM backbone. Hugging Face 공식 경량 VLM.
- **SmolLM-2** (Allal 2025): SmolVLM-2 안의 LLM backbone.
- **LeRobot** (Cadene 2024): SmolVLA의 학습·추론 framework. HF가 만든 robotics SDK.
- **Diffusion Policy** (Chi 2023): Action diffusion의 원조. SmolVLA의 flow matching과 유사하지만 SmolVLA는 VLM 추가.
- **ACT** (Zhao 2023): Action chunking의 시초. SmolVLA가 chunking 그대로 채택.

### 8.2 후속 (본 프로젝트 8편과의 연결)

| 후속 | SmolVLA와의 관계 |
|---|---|
| **[[π0]]** (큰 형) | SmolVLA의 직접적 영감. 다음 정독에서 자세히 |
| **[[π0.5]]** | π0 + open-world generalization. SmolVLA의 작은 data 한계와 대조 |
| **[[π★0.6]]** | π0.5 + RL self-improvement. SmolVLA의 imitation-only 한계 극복 |
| **[[π0.7]]** | π★0.6 + steerable / compositional |
| **[[GR00T-N1]]** | Humanoid + diffusion action head. SmolVLA의 SO-100 한계 → humanoid scale |
| **[[RT-2]]**, **[[OpenVLA]]** | 다른 패러다임 (token-based). 본 모델들과 대조군 |

### 8.3 Architecture-Evolution Tree

SmolVLA는 두 흐름의 교차점:

```
Token-based paradigm:
  RT-1 → RT-2 → OpenVLA
                     ↘
                       SmolVLA ← (lightweight)
                     ↗
Action expert paradigm:
  Diffusion Policy → π0 → π0.5 → π★0.6 → π0.7
                              ↘
                                GR00T N1 (humanoid)
```

**SmolVLA는 "경량 + open + action expert"라는 새 카테고리** 자체를 만듦. 이전 모델들은 [3B-55B 무거움 / token 또는 expert 둘 중 하나만] 중 하나였음.

## 9. 우리 스터디에서 재현·실험 가능한 포인트

### 9.1 재현 가능성

- **Code**: https://github.com/huggingface/lerobot 안에 SmolVLA 통합
- **Weights**: HuggingFace `lerobot/smolvla_base` 등
- **Community data**: 481 datasets 모두 HF에 공개. Appendix A.1에 list
- **Robot hardware**: SO-100/101 3D print 도면 공개 (~$200)
- **재현 난이도**:
  - Full pretrain: 30K GPU-hours, 4 A100 (논문 기준). 또는 single GPU + batch 64로도 가능
  - **Fine-tune**: 100K-200K steps, **single GPU**, 24-48시간 추정
  - **Inference**: **consumer GPU (RTX 4090 16GB) 또는 CPU 가능**

### 9.2 우리 스터디 Track B 진입 경로 (사용자 미래 hands-on 시)

**가장 현실적인 첫 실습 (옵션 B 또는 Track A 완료 후)**:

```bash
# 1. LeRobot install
pip install lerobot

# 2. SmolVLA weights download
from lerobot.models import SmolVLA
model = SmolVLA.from_pretrained("lerobot/smolvla_base")

# 3. (선택) LIBERO sim에서 zero-shot inference
# 4. Single task에 fine-tune (10-50 demonstrations)
# 5. Async inference로 실시간 작동
```

**필요 자원** (Single task LoRA-스타일 FT):
- 1 GPU (RTX 4090 또는 A100, 16-24GB)
- 10-50 demonstrations (LeRobot dataset format)
- 24~48 hours 학습

### 9.3 흥미로운 ablation / new idea 후보 (Track c)

| Idea | 메커니즘 | 기대 효과 | 난이도 |
|---|---|---|---|
| Layer skip N sweep | N ∈ {4, 6, 8, 10, 12, 16} | Sweet spot 정량화 | 낮음 |
| VLM backbone 교체 | SmolVLM-2 → InternVL-tiny, Qwen2-VL-2B 등 | Robotics-friendly VLM 탐색 | 중간 |
| Robotics-PT VLM | OCR 위주가 아닌 robotics-specific 사전학습 | Better features | 높음 |
| Async threshold $g$ sweep | g ∈ {0.3, 0.5, 0.7, 0.9} | Latency vs reactivity | 낮음 |
| Flow matching steps | 5, 10, 20, 50 | Inference time vs quality | 낮음 |
| CA+SA 비율 sweep | 모든 CA / 모든 SA / 1:1 / 2:1 등 | Best interleaving | 중간 |
| Multi-embodiment community data | SO-100 + Franka + Kuka data 모두 학습 | Cross-embodiment generalization | 중간 |
| LoRA on action expert | Frozen VLM + LoRA on $v_\theta$ | 더 efficient FT | 낮음 |
| Flow matching → Diffusion 교체 | Same condition, different generative head | π0과 비교 | 중간 |
| Action chunk hierarchical | n=10 short + n=50 long 두 expert | Long-horizon | 높음 |

### 9.4 LLM 엔지니어 관점 — 한 페이지 요약

SmolVLA = **"frozen VLM(450M) + 작은 flow matching transformer(100M)를 attention으로 연결, action chunk를 생성한다"**.

| 단계 | LLM 엔지니어 작업 비유 |
|---|---|
| 1. SmolVLM-2 backbone | LLaVA-tiny 같은 작은 VLM 사용 |
| 2. Layer skipping | Early exit (Tang 2023). Top layer는 generation용이라 perception에 불필요 |
| 3. Frozen VLM | Frozen base + adapter pattern (LoRA, prefix tuning과 유사) |
| 4. Action expert | Separate small transformer (mixture-of-experts에서 단일 expert) |
| 5. Flow matching | Diffusion model (LLM 도메인에서는 image generation에 익숙) |
| 6. Interleaved CA+SA | Cross-attention은 input conditioning, SA는 token interaction |
| 7. Async inference | Speculative decoding의 robotics 버전 (다른 메커니즘 같은 목표) |
| 8. Community data + VLM auto-annotation | Synthetic data refinement (LLM 분야에서 흔함) |

→ **모든 building block이 LLM/VLM 도메인의 친숙한 도구들의 robotics 적용**. 새로 배워야 할 robotics-specific 지식: flow matching (LLM에서 거의 안 쓰이는, image gen에서 빌려옴) + action chunking 정도.

---

## 부록: 인용 / 추가 자료

### A. 함께 읽기

- **[[π0]]** — SmolVLA의 직접적 영감. **다음 정독 대상 ★**. Flow matching action expert paradigm의 원조.
- **[[OpenVLA]]** — 다른 패러다임의 SOTA open VLA. SmolVLA의 비교 baseline.
- **[[Diffusion-Policy]]** — Action diffusion 원조. Flow matching의 사촌. Phase 2 batch에서 수집 권장.
- **[[ACT]]** — Action chunking 시초. SmolVLA의 chunk=50도 여기서 옴.
- **[[SmolVLM-2]]** — VLM backbone. Marafioti 2025, arXiv:2504.05299.
- **[[Flow-Matching]]** (Lipman 2022, Liu 2022) — 이론 배경. 정독 권장 (π0 정독 직전).

### B. 공식 자료

- Code: https://github.com/huggingface/lerobot (LeRobot framework 안에 SmolVLA)
- Weights: HuggingFace `lerobot/smolvla_base` 등
- Community data: Appendix A.1의 481 datasets 목록
- Robot hardware: SO-100/101 design at https://github.com/TheRobotStudio/SO-ARM100
- Blog: HuggingFace 공식 announcement

### C. 본 요약 작성 중 발견한 핵심 통찰

1. **"action expert paradigm" 첫 등장**: 본 프로젝트 8편 중 SmolVLA가 첫 action expert 모델. π0 정독을 위한 좋은 워밍업.
2. **"frozen VLM" 결정**: OpenVLA의 "unfreeze everything"과 정반대. 두 모델 모두 작동하지만 효율 측면에서 SmolVLA의 frozen이 우월.
3. **Layer skipping의 일반 원리**: VLM의 마지막 N층은 자연어 generation에 특화 → robot 제어엔 불필요. 이는 향후 모든 VLA에 적용 가능한 원칙.
4. **Async inference는 model-agnostic**: SmolVLA에만 적용되는 게 아님. RT-2, OpenVLA, π0에도 적용 가능. **별도 fork project로 가치 있음**.
5. **Community data viability**: 23K episode + 단일 embodiment로도 OpenVLA 7B를 능가 → "더 많은 data = 더 좋은 모델"이라는 통설이 작은 모델에서는 깨질 수 있음.

### D. SmolVLA vs OpenVLA 핵심 비교

| 차원 | OpenVLA (2024) | SmolVLA (2025) |
|---|---|---|
| 패러다임 | **Action token (discrete bin)** | **Flow matching expert (continuous)** |
| Backbone | Llama-2 7B + DINOv2+SigLIP | SmolLM-2 16층 중 8층 + SigLIP (~450M) |
| VLM 학습 | Unfreeze everything | **Frozen** |
| Action expert | 없음 (LLM 자체가 action 출력) | **별도 transformer (~100M)** |
| Generative method | Discrete next-token | **Flow matching (10 step ODE)** |
| Data | 970K OpenX, 22 embodiment | **23K community, ~SO-100 단일** |
| GPU 학습 | 64 A100 14일 | **4 GPU 또는 single GPU** |
| Inference | 6 Hz on 4090 | **~30 Hz async on 4090**, CPU 가능 |
| Multi-camera | No (single image) | **Yes (top, wrist, side)** |
| State input | No (implicit) | **Yes (1 prefix token)** |
| Action chunk | No (1 step) | **Yes (n=50)** |
| LoRA recipe | Yes (Table 1) | Implicit (action expert만 학습) |
| LIBERO Avg | 76.5% | **87.3%** |
| 어느 쪽이 우위? | Pre-training data scale | **Architecture efficiency, smaller data, lighter** |

→ **SmolVLA가 OpenVLA의 패러다임 한계를 인정하고 다른 방향으로 진보**.

### E. 본 정독 후 권장 다음 단계

이론 트랙 (Option C로 결정됨):
1. **다음 정독: π0** — SmolVLA의 큰 형. Flow matching의 원조 robotics 모델. 보유 PDF 있음.
2. π0 정독하면 SmolVLA의 design choice 거의 모두 이해됨 (왜 flow matching, 왜 frozen VLM, 왜 chunk=50)
3. 그 후 π0.5, π★0.6, π0.7로 패러다임 발전 추적
4. 마지막 GR00T N1로 humanoid extension 확인
5. Synthesis 3편 작성 → Track A 종료 → Track B (hands-on)
