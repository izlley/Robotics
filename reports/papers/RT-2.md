# RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control

> **출처**: Brohan et al., Google DeepMind, 2023. arXiv:2307.15818 (28 Jul 2023). Project page: https://robotics-transformer2.github.io
> **읽은 일자**: 2026-05-18
> **PDF**: [`papers/core-models/RT-2-Vision-Language-Action-Models-Transfer-Web-Knowledge-to-Robotic-Control.pdf`](../../papers/core-models/RT-2-Vision-Language-Action-Models-Transfer-Web-Knowledge-to-Robotic-Control.pdf)
> **분량**: 본문 12 페이지 + 부록 14 페이지 = 26 페이지

---

## 한 줄 요약

기존 web-scale VLM(PaLI-X, PaLM-E)을 robot trajectory 데이터에 **co-fine-tune** 하되, **action을 language token처럼 표현**해서 모델 구조 변경 없이 그대로 학습 — 결과적으로 web 지식이 transfer되어 unseen object·symbol·reasoning에서 baseline(RT-1) 대비 2~3배 성능, 다만 새 motor skill은 못 얻음.

## TL;DR

- VLA = VLM의 next-token prediction loss를 그대로 쓰되 "action"이라는 새 token 종류를 추가
- 6-DoF action을 차원당 256 bin으로 discretize한 뒤 9 token sequence로 표현 (termination + 6 axes + gripper... 정확히는 8 token)
- Robot data로 fine-tune할 때 web data를 50~66% 비중으로 함께 흘리는 **co-fine-tuning**이 핵심 (단순 fine-tune 대비 +10pp 이상)
- 새 architecture 없음 → PaLI-X 5B/55B, PaLM-E 12B 그대로 재사용 → web 지식이 자연스럽게 robot 제어로 transfer
- 한계: 새 motor skill 학습 불가, 1~3 Hz inference, closed model — 이 한계들이 OpenVLA, π0, SmolVLA 등 후속 모델의 motivation이 됨

---

## 1. Motivation & 문제 정의

### 1.1 풀려는 문제

**시나리오**: 식탁 위에 빨간 컵, 파란 컵, 사과가 있다. 사용자가 "사과를 빨간 컵에 넣어"라고 말한다.

이 한 문장에 담긴 능력은:
1. "사과"가 무엇인지 visual recognition
2. "빨간"이라는 색 개념
3. "컵"이라는 object category
4. "넣다"가 어떤 motor 행동을 의미하는지
5. 사과를 집어서 컵 안으로 이동시키는 실제 trajectory

(1)~(4)는 web image-text 데이터에서 학습된 VLM(예: PaLI-X)이 잘한다. (5)는 robot teleoperation 데이터에서만 학습 가능. 그런데 이 둘이 따로 학습되면, "빨간 망고를 노란 그릇에"라는 unseen 명령에는 망고 인식은 되어도 motor 동작이 generalize 안 된다.

**연구 질문**: web-scale로 학습된 VLM을 **직접 low-level robot 제어에 통합**할 수 있는가? 그러면 generalization과 reasoning이 자연스럽게 robot에서도 emergent하게 나타날 것인가?

### 1.2 기존 방법의 한계

| 접근 | 사례 | 한계 |
|---|---|---|
| LLM/VLM = high-level planner only | SayCan, PaLM-E (원본), ChatGPT for Robotics | LLM은 "사과를 집어라"까지만 출력. 실제 motor command는 별도 low-level controller가 담당 → low-level 단계에서는 web 지식 활용 못함 |
| 새 VLA 아키텍처 처음부터 | Gato (Reed 2022) | 거대한 web pretraining 자산을 못 씀. Robot data scale에 갇힘 |
| 작은 VLM을 perception aux로만 | CLIPort, MOO | VLM이 "객체 mask" 같은 보조 정보만 제공. 정책 자체는 작은 transformer에 의존 |
| Visual representation pretrain만 | R3M, VC-1 | Visual encoding은 풍부해도 language reasoning은 약함 |

공통 한계: **VLM의 entire weight space를 정책으로 그대로 활용한 적이 없었다**. 마치 LLM을 코드 작성에 쓰면서 "LLM은 자연어 처리만 하고, 코드는 별도 코더가 받아쓰게 하자"는 식.

### 1.3 본 논문의 가설

> "Vision-language model을 그대로 직접 fine-tune하여 action을 출력시킬 수 있다. 단, action은 새 modality가 아니라 **기존 text vocabulary를 빌려쓰는 token sequence**로 표현하면 된다."

이 가설이 맞다면:
- 새 architecture·parameter·optimizer 변경 불필요
- VLM의 pretrained weights와 capability가 그대로 robot 정책에 흘러들어옴
- "사과 = apple = 林檎 = manzana" 같은 multilingual·semantic concept이 robot generalization으로 자연스럽게 이어짐

## 2. 핵심 아이디어

### 2.1 한 줄

**Action도 text token이다.** 6-DoF robot action을 정수 ID sequence로 discretize한 뒤 VLM tokenizer의 기존 token slot에 매핑하면, VLM은 새 파라미터 추가 없이 standard language modeling loss로 fine-tune된다.

### 2.2 왜 직관과 어긋나는가

LLM/VLM 엔지니어의 자연스러운 직관은:
- "Action은 continuous → regression head를 따로 둬야"
- "Robot은 정밀한 결정 = continuous control" → discrete token으로는 부족할 것

RT-2는 정반대 가설을 검증:
- "256 bin이면 충분히 정밀" (manipulation은 1 cm 수준이면 됨)
- "Discrete token + LM loss로도 emergent generalization이 더 강력하다"
- "Regression head는 web pretraining 자산을 못 받음"

이 trade-off가 후속 모델 흐름의 분기점이 된다 (3.5 절 참조).

### 2.3 LLM/VLM 도구와의 analogy

| LLM 개념 | RT-2 대응 |
|---|---|
| Domain-specific token 추가 (`<eos>`, `<|fim_start|>`) | 256개 action token 추가 (least-frequent vocab 위에 overwrite) |
| Multi-task instruction tuning | Web VQA + robot trajectory를 한 mixture에 |
| Constrained decoding (JSON mode, grammar-constrained) | Robot prompt 시 action vocab으로만 sampling 제한 |
| Chain-of-Thought prompting | `Plan: ...자연어... Action: ...token...` 형식 학습 |
| Catastrophic forgetting 방지 | Co-fine-tuning (web data 50~66% mix) |
| Few-shot in-context learning | RT-2에서는 없음 — closed-loop 제어라 zero-shot 추론 |
| Tokenizer vocab size 50K~256K | PaLI-X 256K급, action에 256 slot만 할당 |

**핵심 통찰**: RT-2는 새로운 "robotics ML 기법"이 아니다. **LLM/VLM에서 이미 검증된 도구들의 robotics 응용**일 뿐이다.

## 3. 아키텍처 (상세)

### 3.1 입력 / 출력

| 항목 | 형식 | 차원 | 예시 |
|---|---|---|---|
| Vision | RGB image (1장) | (H, W, 3) | 카메라 frame |
| Language prompt | 자연어 + 지정 형식 | text string | `"Q: what action should the robot take to pick up the strawberry? A:"` |
| Robot state / proprioception | **명시적 입력 없음** | — | 이미지에서 implicit하게 학습 |
| Output | 8 정수 token (공백 구분) | 8개 token ID | `"1 128 91 241 5 101 127 142"` |

**중요**: RT-2는 robot state(joint angle 등)를 명시적 입력으로 받지 않는다. 카메라 이미지 한 장 + 자연어 명령만으로 다음 action을 결정. 이는 단순하지만, 손이 안 보이는 occlusion이나 force feedback이 필요한 task에서 약점이 된다.

### 3.2 Backbone

논문은 **2개 + 1개**의 instantiation을 평가:

| 모델 이름 | Vision | Multimodal Backbone | 전체 파라미터 |
|---|---|---|---|
| **RT-2-PaLI-X-5B** | ViT-22B share* | UL2-style enc-dec, 32B params, 50 layers | 5B (downscaled) |
| **RT-2-PaLI-X-55B** | ViT-22B (Dehghani 2023) | UL2-style enc-dec, 32B params, 50 layers | 55B |
| **RT-2-PaLM-E-12B** | ViT-4B | PaLM decoder-only | 12B |
| RT-2-PaLI-3B (보조) | ViT-G/14 (2B) | UL2-3B | 3B (Language-Table 시뮬 실험용) |

(*) 5B와 55B는 같은 ViT를 공유하고 backbone 크기만 다름.

**구조적 차이**:
- PaLI-X: encoder-decoder (T5/UL2 계열). Image+text가 encoder에, action token이 decoder 출력
- PaLM-E: decoder-only. Image features를 language embedding space에 projection한 뒤 일반 LLM처럼 처리

두 backbone 모두 next-token prediction objective로 web에서 pretraining된 상태. RT-2는 이 weight를 시작점으로 robot data + web data를 mix해서 fine-tune.

### 3.3 핵심 모듈: Action Tokenization

이게 RT-2의 진짜 핵심이다. 메커니즘을 단계별로:

**Step 1. Action vector 정의**

Robot end-effector에 대해 다음 8차원 vector를 매 step 출력해야 한다:

```
action = [terminate, Δpos_x, Δpos_y, Δpos_z, Δrot_x, Δrot_y, Δrot_z, gripper_extension]
```

- `terminate ∈ {0, 1}`: episode 종료 여부 (discrete)
- `Δpos_x, y, z ∈ [-Δ_max, +Δ_max]`: end-effector 위치 delta (continuous, 메터 단위)
- `Δrot_x, y, z ∈ [-θ_max, +θ_max]`: end-effector 회전 delta (continuous, radian)
- `gripper_extension ∈ [0, 1]`: gripper 개폐 정도 (continuous, 0=close, 1=open)

총 1개 discrete + 7개 continuous = 8 차원.

**Step 2. Continuous 차원을 256 bin으로 discretize**

각 continuous 차원에 대해 균등 분할:

$$
\text{bin}(a) = \left\lfloor \frac{a - a_{\min}}{a_{\max} - a_{\min}} \times 256 \right\rfloor
$$

예시: Δpos_x ∈ [-0.05, +0.05] (m) 범위라면:
- bin 0 → -0.05 m
- bin 128 → 0.0 m (중앙)
- bin 255 → +0.05 m
- bin width = 0.1 m / 256 ≈ **0.39 mm** (충분히 미세)

이렇게 7개 continuous 차원을 각각 256 bin으로 → 8개 bin index 출력 가능.

**Step 3. Bin index를 VLM tokenizer의 token ID로 매핑**

VLM의 tokenizer는 보통 32K~256K vocabulary를 가진다. 그 중 256개 slot을 빌려야 한다. **방법은 backbone마다 다름**:

| Backbone | 매핑 방법 | 왜? |
|---|---|---|
| **PaLI-X** | 정수 0~999 각각 unique token이 이미 있음 → bin index i를 token "i"로 그대로 매핑 | PaLI-X는 multilingual 수치 처리를 위해 정수 token을 미리 분리해뒀음 |
| **PaLM-E** | Vocabulary 빈도 통계 상 **least frequently used 256개 token을 overwrite** → 그 token slot의 의미를 "action bin"으로 재학습 | PaLM-E는 그런 편의가 없음. Symbol tuning (Wei 2023) 그대로 활용 |

**Symbol tuning이 뭔가?** LLM에서 anchor word(예: "positive"/"negative")를 임의 symbol(예: "foo"/"bar")로 바꿔 학습하면, 모델이 in-context 매핑을 더 robust하게 학습한다는 발견. RT-2는 이를 응용: 거의 안 쓰이는 token을 "action bin index"라는 새 의미로 재학습.

**Step 4. 최종 출력 string**

8 token을 공백으로 join한 단일 string:

```
"1 128 91 241 5 101 127 142"
 │  │   │  │   │  │   │   └── gripper bin
 │  │   │  │   └──┴───┴────── 회전 3축 bin
 │  └───┴──┴───────────────── 위치 3축 bin
 └──────────────────────────── terminate flag
```

VLM 입장에서는 그냥 "8개 정수의 sequence"를 next-token prediction하는 것일 뿐. **새 head·loss·layer 추가 0**.

### 3.4 Decoding 시 Output Constraint

Inference 시점에 한 가지 트릭이 더 있다. VLM은 원래 자유로운 자연어 출력이 가능한데, robot prompt가 들어왔을 때 자연어로 답하면 안 된다. 그래서:

```python
# pseudocode
def decode(prompt, image):
    if is_robot_prompt(prompt):
        allowed_vocab = action_token_set    # 256 + termination 등
    else:
        allowed_vocab = full_vocab          # web task일 때는 전체
    return vlm.generate(prompt, image, vocab_mask=allowed_vocab)
```

이는 LLM의 **constrained decoding** (JSON mode, grammar-constrained decoding)과 본질적으로 같다. Token sampling 시 logit mask로 비-action token의 확률을 0으로 강제.

### 3.5 전체 다이어그램

```
┌──────────────────────────────────────────────────────────────┐
│  Input                                                       │
│  ┌─────────────┐   ┌──────────────────────────────────────┐ │
│  │  RGB image  │   │ "Q: what action should the robot     │ │
│  │             │   │   take to pick up the strawberry? A:"│ │
│  └──────┬──────┘   └────────────────┬─────────────────────┘ │
└─────────┼───────────────────────────┼───────────────────────┘
          │                           │
          ▼                           ▼
     ┌─────────┐                ┌──────────┐
     │  ViT    │                │ Tokenizer│
     │  (22B)  │                │          │
     └────┬────┘                └────┬─────┘
          │ image tokens             │ text tokens
          └────────────┬─────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  VLM Backbone                    │
        │  (PaLI-X 32B / PaLM 12B)         │
        │  → next-token prediction         │
        │  → constrained to action vocab   │
        └────────────────┬─────────────────┘
                         │
                         ▼
        Output token string: "1 128 91 241 5 101 127 142"
                         │
                         ▼ de-tokenize (bin → real value)
        action = [
          terminate = 1,
          Δpos = (-0.001, -0.034, +0.062),   # m
          Δrot = ( 0.022,  0.005, -0.041),   # rad
          gripper = 0.55
        ]
                         │
                         ▼
        ┌──────────────────────────────────┐
        │  Robot controller (closed-loop)   │
        │  실행 frequency: 1~5 Hz           │
        │  (55B model = 1~3 Hz)            │
        └──────────────────────────────────┘
```

### 3.6 구체 예시 — Training data sample

훈련 데이터 한 sample은 다음과 같이 보인다:

**Input (multimodal)**:
- Image: kitchen 카운터 위 strawberry, bowl, sponge가 보이는 frame
- Text: `"Q: what action should the robot take to pick up the strawberry? A:"`

**Target (text)**:
- String: `"1 128 91 241 5 101 127 142"` (= 8 token sequence)

Loss는 그냥 standard NLL over 이 8 token. **Robot data든 web data든 형식이 동일**:

| Sample type | Image | Text Prompt | Target |
|---|---|---|---|
| Robot | kitchen frame | "Q: what action ... pick strawberry? A:" | "1 128 91 241 ..." |
| VQA | dog 사진 | "Q: what color is the dog? A:" | "brown" |
| Captioning | 일몰 사진 | "Caption:" | "A beautiful sunset over the ocean" |

→ VLM은 자신이 robot data를 보고 있는지 captioning을 보고 있는지 **구조적으로 구분 안 함**. Loss 동일, 학습 알고리즘 동일.

## 4. 데이터 (상세)

### 4.1 Web 데이터 (co-fine-tuning에 함께 흘리는 부분)

**WebLI** (PaLI-X 원본 학습 데이터의 핵심):
- 원본: **10B image-text pair**, 109개 언어
- Filtering: cross-modal similarity score (CLIP-style)로 상위 10% → **1B pair**
- 추가 데이터셋: VQA 데이터(VQAv2, OK-VQA 등), captioning(COCO Captions 등)

**스케일 감각** (LLM 엔지니어용):
- 1B image-text pair ≈ LLaMA-2 학습량(2T text token)의 약 1/2000~1/200 수준 (image당 평균 token 수에 따라)
- 그러나 multimodal이라 token 수만으로 비교는 부정확
- 핵심은 **모든 web 지식이 이미 PaLI-X / PaLM-E weight에 압축되어 있음** → RT-2는 그걸 재활용

### 4.2 Robot 데이터

RT-1 (Brohan 2022) 데이터셋을 그대로 사용:
- **13개 mobile manipulator** (같은 종류의 robot 13대)
- **17개월간** 수집
- 환경: Google office의 **kitchen**
- 각 episode = teleoperator가 robot을 직접 조작한 trajectory
- **자연어 instruction이 episode마다 annotation**
- 7가지 skill primitive:
  1. PickObject (집기)
  2. MoveObject Near Object (어떤 물체 근처로 이동)
  3. PlaceObject Upright (세워 놓기)
  4. KnockObject Over (쓰러뜨리기)
  5. OpenDrawer (서랍 열기)
  6. CloseDrawer (서랍 닫기)
  7. PlaceObject into Receptacle (용기에 넣기)
  8. PickObject from Receptacle and place on counter (꺼내서 카운터에 놓기)

**Episode 수**: 논문 본문에 명시 없음. RT-1 논문 기준 약 130K episode.

**비교 — LLM 데이터와의 격차**:
- Web text: ~10^12 token
- Robot trajectory: ~10^5 episode × 평균 50 step × 8 token = **4 × 10^7 action token**
- → **약 25,000배 차이**. 그래서 sampling weight를 키워야 함.

### 4.3 데이터 mixture·sampling

훈련 배치(batch) 안에서 robot 데이터와 web 데이터를 섞는 비율:

| 모델 | Robot 비중 |
|---|---|
| RT-2-PaLI-X | **~50%** |
| RT-2-PaLM-E | **~66%** |

수학적으로, batch sampling 확률:
$$
P(\text{sample from robot}) = w_{\text{robot}}, \quad P(\text{sample from web}) = 1 - w_{\text{robot}}
$$
$$
w_{\text{robot}} \in \{0.5, 0.66\}
$$

자연스러운 비율 (데이터 크기 비례)은 robot이 0.004% 정도일 텐데, 의도적으로 ~150~250배 oversample한 셈.

**왜 의도적 oversample?** Robot capability를 학습할 만큼 robot data가 batch에 빈번히 보여야 함. 그렇다고 robot 비율을 너무 높이면(예: 90%) web 지식이 시들해짐. 50~66%가 sweet spot.

### 4.4 Episode 데이터 형식

각 episode는 다음과 같이 풀린다 (실제 training sample 생성 과정):

```
Episode = [(image_0, instr, action_0), (image_1, instr, action_1), ..., (image_T, instr, action_T)]
   ↓ flatten
Samples = [(image_t, instr, action_t)] for all t

각 sample:
  input  = image_t + format(instr)
  target = format(action_t)
```

**중요**: 각 time step이 독립적인 (input, target) sample. History를 명시적 input으로 받지 않음 → markovian assumption. 단, ViT가 보는 image에 이미 robot arm·물체 위치가 시각적으로 포함되어 있어 implicit context는 있음.

## 5. 학습 (상세)

### 5.1 Loss — Next-token prediction

전체 loss는 단순 NLL (Negative Log Likelihood) over token sequence:

$$
\mathcal{L}(\theta) = -\mathbb{E}_{(x, y) \sim D} \left[ \sum_{t=1}^{|y|} \log p_\theta(y_t \mid y_{<t}, x) \right]
$$

기호 정의:
- $\theta$: VLM 전체 parameter (frozen 없음, all-finetune)
- $x = (\text{image}, \text{text prompt})$: multimodal 입력
- $y = (y_1, y_2, \ldots, y_{|y|})$: target token sequence
  - Robot sample: $|y| = 8$ (action token 8개)
  - VQA sample: $|y|$ 가변 (자연어 답변 길이)
  - Captioning: $|y|$ 가변
- $D$: 학습 데이터 분포 (robot + web mixture)
- $p_\theta(y_t \mid y_{<t}, x)$: 이전 token과 입력이 주어졌을 때 t번째 token의 확률 (autoregressive)

### 5.2 Co-fine-tuning loss를 수식으로

논문은 명시적 수식을 안 줬지만, mixture sampling은 다음과 같이 풀린다:

$$
\mathcal{L}_{\text{co-ft}}(\theta) = w_{\text{robot}} \cdot \mathcal{L}_{\text{robot}}(\theta) + (1 - w_{\text{robot}}) \cdot \mathcal{L}_{\text{web}}(\theta)
$$

여기서:
- $\mathcal{L}_{\text{robot}}(\theta) = -\mathbb{E}_{(x, y) \sim D_{\text{robot}}} [\log p_\theta(y \mid x)]$
- $\mathcal{L}_{\text{web}}(\theta) = -\mathbb{E}_{(x, y) \sim D_{\text{web}}} [\log p_\theta(y \mid x)]$
- $w_{\text{robot}} \in \{0.5, 0.66\}$

이는 **multi-task learning의 standard weighted loss**다. 새 알고리즘이 아니라 비율 결정만 했다.

### 5.3 학습 단계

**단일 단계의 co-fine-tuning**. Pretraining → fine-tuning stage가 분리되어 있지 않다.

- VLM 자체의 pretraining은 이미 PaLI-X / PaLM-E 원논문에서 끝남
- RT-2 단계 = "co-fine-tuning"만 (web data + robot data를 mix해서 한 번에)

이게 단순하지만 효과적. LLM 도메인의 SFT (Supervised Fine-Tuning) 단계와 1:1 대응.

### 5.4 Hyperparameter

| 모델 | LR | Batch | Steps | LR schedule |
|---|---|---|---|---|
| RT-2-PaLI-X-55B | 1e-3 | 2048 | **80K** | PaLI-X 원논문 따름 |
| RT-2-PaLI-X-5B | 1e-3 | 2048 | **270K** | 동상 |
| RT-2-PaLM-E-12B | 4e-4 | 512 | **1M** | PaLM-E 원논문 따름 |
| RT-2-PaLI-3B (Lang-Table) | 1e-3 | 128 | 300K | — |

**LLM 엔지니어 시각**: 
- 1e-3 LR은 일반 SFT(2e-5~5e-5)보다 훨씬 큼. 이건 PaLI-X 원논문의 fine-tune lr이 원래 큰 것 (UL2 계열 특성)
- 55B를 80K step만으로 완료 → robot data가 사실상 작아서 빨리 수렴
- PaLM-E-12B는 1M step으로 가장 길게 → 더 작은 backbone을 더 오래 = bias-variance trade-off

### 5.5 Scaling 관찰 (정량)

논문의 Table 6 결과 (Unseen Average %, 단위 = success rate):

| 학습 방식 | 5B | 55B | 5B vs 55B |
|---|---|---|---|
| **from scratch** (VLM pretraining 무시) | **9%** | (skip — 5B에서도 처참) | — |
| **fine-tuning only** (robot data만) | 42% | 52% | +10pp |
| **co-fine-tuning** (web + robot mix) | 44% | **63%** | +19pp |

해석:
1. **VLM pretraining이 결정적**: scratch 9% → fine-tune 42% (5B 기준). **+33pp** 개선의 거의 전부는 web pretraining 덕분
2. **Co-fine-tune > fine-tune**: 5B에서 +2pp, 55B에서 +11pp. **모델 클수록 차이 증폭** — 큰 모델이 web 지식을 더 잘 retain
3. **모델 크기**: 5B → 55B에서 +19pp (co-FT 기준). Scaling law는 robot domain에서도 작동

이 3 axis (pretraining / co-FT / size)가 RT-2 contribution의 정량적 뼈대.

## 6. 평가 (상세)

### 6.1 Setup

- **Robot**: 7-DoF mobile manipulator (RT-1과 동일)
- **환경**: real-world office kitchen + 일부 unseen 환경
- **Trial 수**: 약 **6,000회** (논문 통틀어)
- **비교 방식**: A/B testing framework (모든 baseline을 동일 조건에서 1대1 비교)

### 6.2 Baselines

| Baseline | 종류 | 특징 | 비교 의미 |
|---|---|---|---|
| **RT-1** | Transformer policy, 35M params | VLM pretraining 없음, RT-2와 동일 robot data | "VLM이 안 들어가면 어떻게 되나" |
| **VC-1** | Visual foundation model | ViT-L pretrain만 활용. Language는 USE(Universal Sentence Encoder)로 따로 concat | "visual rep만으로는 부족한가" |
| **R3M** | Visual-language rep | Ego4D(인간 영상) 데이터로 학습한 rep | "다른 visual rep도 비슷한가" |
| **MOO** | VLM-aided perception | VLM이 객체 위치를 색칠한 pixel hint로 알려줌, 정책은 RT-1 | "VLM을 부분적으로만 쓸 때" |

모든 baseline이 **같은 robot data로 학습**됨. 차이는 VLM/visual rep 활용 방식뿐.

### 6.3 핵심 결과 1 — Overall Performance

논문 Table 4. 숫자는 success rate (%):

| Model | Seen Tasks | Unseen Obj (Easy/Hard) | Unseen BG (E/H) | Unseen Env (E/H) | **Unseen Avg** |
|---|---|---|---|---|---|
| R3M | 45 | 32 / 14 | 13 / 9 | 0 / 2 | 12 |
| VC-1 | 63 | 34 / 10 | 13 / 3 | 0 / 0 | 10 |
| RT-1 | 92 | 31 / 43 | 71 / 9 | 26 / 14 | 32 |
| MOO | 75 | 58 / 48 | 38 / 41 | 19 / 3 | 35 |
| **RT-2-PaLI-X-55B** | 91 | 70 / 62 | 96 / 48 | 63 / 35 | **62** |
| **RT-2-PaLM-E-12B** | 93 | 84 / 76 | 75 / 71 | 36 / 33 | **62** |

해석:
- **Seen tasks**: RT-1과 RT-2가 비등 (91~93% vs 92%) — robot data 학습 자체는 RT-1도 잘함
- **Unseen avg**: RT-2 두 변형 모두 **62%**, baseline 최고가 35% → **약 2배** 성능
- **시나리오별 강점**:
  - PaLI-X-55B: Unseen Background에서 96% (Easy) — visual diversity 강함
  - PaLM-E-12B: Unseen Object에서 84% (Easy), 76% (Hard) — semantic naming 강함
- **약한 영역**: Unseen Environment (Hard) — 둘 다 33~35%. **새 환경 자체는 어려움**

### 6.4 결과 해석 — 수치가 의미하는 바

"62% vs 32%"가 2배라는 것은 단순히 더 좋다는 게 아니다. **fail rate 관점**: RT-1은 68% 실패, RT-2는 38% 실패 — **실패율 절반 감소**. 실패가 robot wear, reset cost, downtime을 의미한다는 점에서 산업적으로 큰 차이.

또한 **Seen에서는 비슷**한데 **Unseen에서만 크게 갈린다**는 점이 핵심. 이는:
- RT-2가 robot data 안의 task를 외운 것이 아님 (overfitting 아님)
- web 지식이 새 상황으로의 **generalization gap을 메꾸는 작용** 명확

### 6.5 핵심 결과 2 — Emergent Capabilities

논문 Table 5. Symbol/Reasoning/Person 세 카테고리 평균 success rate (%):

| Model | Symbol Avg | Reasoning Avg | Person Avg | **Overall Avg** |
|---|---|---|---|---|
| VC-1 | 11 | 13 | 13 | 11 |
| RT-1 | 16 | 28 | 20 | 17 |
| **RT-2-PaLI-X-55B** | **82** | 58 | 53 | **60** |
| RT-2-PaLM-E-12B | 36 | 35 | 43 | 40 |

해석:
- **PaLI-X-55B vs RT-1**: 60% vs 17% → **3.5배**
- PaLI-X는 symbol/recognition 강함 (visual VLM이라 합리적)
- PaLM-E는 math reasoning에서 한 발 우위 (PaLM 계열 수학 pretraining)
- → **VLM backbone의 pretraining mixture가 직접적으로 emergent capability를 결정**

**Example tasks**:
- Symbol: "move coke can near 3" (숫자 3이라는 글자/그림 옆으로) — robot data에 절대 없음
- Reasoning Math: "move banana near the sum of two plus one" → 3을 의미
- Reasoning Logos: "move cup to google" (Google logo 옆) — robot data 무관
- Person: "move coke can to person with glasses" — CelebA-style identity 분별
- Multilingual: "mueve la manzana al vaso verde" (스페인어), "déplacer les frites verts dans la tasse rouge" (프랑스어)

이 모든 task에서 robot training 데이터에는 해당 개념·표현이 **존재하지 않는다**. 그럼에도 작동 → web pretraining이 그대로 transfer.

### 6.6 Ablation — Co-fine-tuning vs Fine-tuning vs Scratch

(5.5에 이미 표 있음, 여기서 재인용 + 메커니즘 해석)

| 학습 | 5B Unseen Avg |
|---|---|
| from scratch | 9% |
| fine-tuning | 42% |
| co-fine-tuning | 44% |

**왜 co-fine-tune이 fine-tune보다 좋은가?** 메커니즘:

1. **Forgetting 관점**: Pure fine-tune은 robot loss만 minimize → web pretraining에서 형성된 weight가 robot loss 방향으로 끌려간다 → web에서 알던 "사과 = apple = 林檎"이 점점 무뎌진다 → unseen object generalization 떨어진다.

2. **Implicit regularization 관점**: Web data가 mixture에 있으면 매 step gradient가 "robot loss 줄이기 + web loss 유지하기"의 절충안 — 거대 weight space에서 **robot-only solution이 아니라 web-and-robot joint solution**으로 수렴.

3. **Effective data augmentation 관점**: Web image-caption pair는 매우 다양 (수십억 종) — robot data의 visual distribution을 implicit하게 확장.

### 6.7 Chain-of-Thought 변형

RT-2-PaLM-E를 few hundred steps 더 fine-tune하면서 데이터 augmentation:

**Before** (일반 RT-2):
```
Input:  "Q: what action ... pick rxbar chocolate? A:"
Target: "1 128 124 136 121 158 111 255"
```

**After** (CoT RT-2):
```
Input:  "Instruction: I'm hungry."
Target: "Plan: pick rxbar chocolate. Action: 1 128 124 136 121 158 111 255"
```

자연어 plan을 먼저 출력하고 그 다음 action token. **본질적으로 LLM의 CoT prompting을 robot에 그대로 이식**.

결과 (정성적): 
- "I'm hungry" → Plan: pick rxbar chocolate → Action
- "I need to hammer a nail, what's useful?" → Plan: pick rock → Action
- 다단계 indirection 가능 (직접 robot data에 없는 reasoning)

**구조적 의미**: VLM이 action token 출력 직전에 "intermediate text token"을 생성할 시간을 갖는다. LLM의 CoT가 "더 많은 compute을 reasoning에 투입"하는 효과와 동일.

### 6.8 정성적 발견 — Failure Cases

논문에서 인정한 실패 패턴:

| 실패 유형 | 예시 | 추정 원인 |
|---|---|---|
| Object dynamics unseen | pen이 굴러 떨어짐, banana center-of-mass 잘못 잡음 | Robot data가 block/can 등 잡기 쉬운 물체에 편향 |
| Grasping by specific parts | "handle을 잡아라" 명령 어려움 | Robot data가 부위 명세 적음 |
| Novel motor primitives | wiping, towel folding | Web data에서 motor skill은 학습 안 됨 |
| Dexterous/precise | folding | Discretization 정밀도 한계? 더 큰 원인은 데이터 부족 |
| Multi-step indirection | "X에 사용할 도구를 찾고, 그것으로 Y를 해라" | CoT length 한계 |

## 7. 강점 / 한계

### 7.1 강점

| 강점 | 구조적 원인 |
|---|---|
| Web 지식 transfer | VLM 전체를 통째로 fine-tune → semantic·visual·multilingual concept이 robot에 그대로 흘러옴 |
| 단순함 | 새 layer·loss·optimizer 추가 없음. Standard LM training이 그대로 적용 |
| Pretraining amortize | PaLI-X 1B image-text 학습 비용은 이미 지불 완료 → robotics가 무료 활용 |
| CoT 결합 자연스러움 | Text token으로 action 표현 → 자연어 plan token이 같은 sequence에 들어갈 수 있음 |
| Backbone-agnostic | PaLI-X든 PaLM-E든 동작 → 미래의 더 좋은 VLM이 나오면 그대로 갈아끼우면 됨 |
| Symbol·Reasoning emergent | Web에서 학습된 추상 개념(숫자, 인물, 로고, 언어)이 robot 명령에 직접 사용 가능 |

### 7.2 한계 — Mechanism 분석

| 한계 | 구조적 원인 | 후속 모델은 어떻게 해결? |
|---|---|---|
| **새 motor skill 학습 불가** | Web data는 "이게 사과다"는 알지만 "사과를 어떻게 집는지"는 없음. Robot data scale이 작아서 skill 분포가 제한됨 | π0 시리즈: 더 큰 robot dataset + RL self-improvement. GR00T: synthetic data |
| **Inference latency 1~3 Hz** | 55B 모델은 cloud TPU 필요. High-freq control(50+ Hz) 불가능 | SmolVLA: 450M 경량화. π0-FAST: action chunking으로 step수 줄임 |
| **Closed model** | PaLI-X, PaLM-E 모두 비공개 | OpenVLA: Llama-2 7B 오픈 backbone 사용 |
| **Discrete action 정밀도 ceiling** | 256 bin은 manipulation OK, dexterous에는 부족 | π0: flow matching으로 continuous output. Diffusion Policy: diffusion으로 trajectory 생성 |
| **Single embodiment** | Robot data가 한 종류 로봇 | X-VLA: soft prompt로 cross-embodiment. RT-X: 22개 robot 통합 |
| **No state input** | Image만 사용, joint angle 등 명시 안 함 | π0.5: state encoder 추가. 일부 모델은 force/tactile까지 |
| **Action chunking 없음** | 매 step 1 action — 50step trajectory면 50번 inference | ACT: 한 번에 k action 출력. π0: action chunk |
| **No memory** | Markovian — 직전 step 기억 없음 | π0.7: memory module. World model 기반 접근 |

이 한계 표가 **본 프로젝트 후속 정독 8편의 거의 전부의 motivation을 설명**한다.

## 8. 다른 모델과의 관계

### 8.1 직접적 선행 연구

- **RT-1** (Brohan 2022, arXiv:2212.06817):
  - 35M params Transformer-based robot policy
  - VLM pretraining 없이 robot data만
  - **Action discretization 256-bin scheme은 RT-1이 원조**
  - **Robot dataset 그대로 RT-2가 재사용**
- **PaLI-X** (Chen 2023a): RT-2-PaLI-X의 backbone. 5B / 55B 두 size
- **PaLM-E** (Driess 2023): RT-2-PaLM-E의 backbone. 12B. PaLM-E 자체도 robotics multimodal model이지만 high-level planning 출력
- **Symbol tuning** (Wei 2023): RT-2-PaLM-E가 least-frequent vocab을 action으로 overwrite하는 정당화 근거

### 8.2 후속 (본 프로젝트 8편과의 연결)

| 후속 모델 | RT-2와의 관계 |
|---|---|
| **OpenVLA** (Stanford/UCB 2024) | RT-2의 오픈 버전. Llama-2 7B + DINOv2/SigLIP. 같은 "action as text token" 패러다임. 우리 다음 정독 ✅ |
| **π0** (PI 2024) | RT-2의 **반대 방향**. Discrete token 버리고 flow matching action expert를 별도로 부착. VLM은 perception/planning만, action은 별도 head |
| **π0.5** (PI 2025) | π0 + open-world generalization. RT-2의 co-fine-tuning idea 확장 |
| **π★0.6** (PI 2026) | π0.5 + RL self-improvement. RT-2의 "static policy" 한계 극복 |
| **π0.7** (PI 2026) | π★0.6 + steerable / compositional |
| **SmolVLA** (HF/LeRobot 2025) | RT-2 latency 한계 직격. 450M 경량화 |
| **GR00T N1** (NVIDIA 2025) | Humanoid 확장. RT-2 패러다임 + diffusion action head |
| **X-VLA** / **Octo** | Cross-embodiment. RT-2의 single robot 한계 해결 |

### 8.3 Architecture-Evolution Tree

```
        RT-1 (35M, transformer policy, no VLM)
          │ + VLM web pretraining
          ▼
        RT-2 (5B/12B/55B, action as text token)          ← 여기 ★
          │
          ├── + 오픈소스화 ────────────→ OpenVLA (7B)
          │
          ├── + continuous action expert ──→ π0 → π0.5 → π★0.6 → π0.7
          │       (flow matching, diffusion)
          │
          ├── + 경량화 (on-device) ──────→ SmolVLA (450M)
          │
          ├── + humanoid / sim data ─────→ GR00T N1
          │
          ├── + cross-embodiment ────────→ X-VLA, Octo
          │
          └── + tactile/force ────────────→ Rho-alpha
```

**중요한 관찰**: RT-2는 단일 줄기에서 **부채꼴로 후속이 갈라진다**. 어떤 후속도 RT-2의 모든 측면을 한 번에 개선하지 않고, 각자 한 axis씩 골라 발전. → 후속 정독은 "어느 axis인가"를 의식하며 보면 효율 ↑

## 9. 우리 스터디에서 재현·실험 가능한 포인트

### 9.1 재현 가능성

- **RT-2 자체**: PaLI-X / PaLM-E weights 비공개 → 재현 불가
- **대안 1**: **OpenVLA** (다음 정독) — Llama-2 7B + 공개 코드 + 공개 weight + Open X-Embodiment data
- **대안 2**: Action tokenization 자체는 공개 → 자체 VLM에 이식 가능 (예: Llava + 256 bin action tokenizer)
- **대안 3**: HF SmolVLA stack에서 RT-2 스타일 token-based action을 다시 구현하는 ablation

### 9.2 흥미로운 ablation / new idea 후보 (Track c 단계용)

본 프로젝트 학습 절차 (c) "new idea 실험" 단계에서 시도해볼 만한 것들 (체계적으로 정리):

| Idea | 메커니즘 | 기대 효과 | 난이도 |
|---|---|---|---|
| Co-FT ratio sweep | $w_{\text{robot}} \in \{0.3, 0.5, 0.66, 0.8, 0.9\}$ | Sweet spot 정량화 | 낮음 |
| Action vocab slot 선택 | Random / least-frequent / numeric-shaped / freshly-added token slot 비교 | Symbol tuning의 robustness 검증 | 낮음 |
| Bin 수 sweep | 64 / 128 / 256 / 512 / 1024 bin | Dexterous task에서 ceiling 확인 | 낮음 |
| Output constraint 효과 | Decode 시 vocab mask vs no mask | Constraint 없을 때 어떤 failure? | 매우 낮음 |
| CoT length scaling | Plan 자연어 길이 0, 10, 30, 100 token | Reasoning task 성공률 곡선 | 중간 |
| Action token ordering | termination flag 위치, 차원 순서 permutation | NLL 학습 효율, autoregressive bias | 낮음 |
| Image scale-up | 1 image → multi-view (3개 카메라) | Occlusion 극복 | 중간 (데이터 필요) |
| State input 추가 | proprioception 명시 input | Force feedback 필요 task | 중간 |
| Action chunking | 1 step → k step (ACT 스타일) | Latency / smoothness | 중간 |
| Continuous head 교체 | Token → flow matching head | π0 vs RT-2 직접 비교 | 높음 |

### 9.3 Stack 호환성

- **OpenVLA repo**: RT-2와 동일 패러다임이므로 코드 즉시 이식 가능
- **LeRobot**: RT-2 직접 지원은 없지만 SmolVLA / OpenVLA 통한 우회 가능
- **openpi (Physical Intelligence)**: 다른 패러다임 — 직접 호환 안 됨. 비교 실험에 유용

### 9.4 LLM 엔지니어 관점 — 한 페이지 요약

RT-2를 한 문장으로: **"PaLI-X/PaLM-E를 robot trajectory data + web data로 SFT하되, action을 token으로 표현했다."**

이 한 문장을 풀면:

| 단계 | LLM 엔지니어 작업 비유 |
|---|---|
| 1. Action discretize | LLM에 sentiment 분류시 `1,2,3,4,5` 같은 정수 label로 답하게 학습 |
| 2. Vocab slot 할당 | Special token `<sentiment_1>` 같은 거 vocab에 추가하는 것 |
| 3. Co-FT | LIMA 류 SFT에서 base PT data 일부를 함께 흘리는 것 |
| 4. Constrained decode | JSON mode / logit_bias 사용하는 것 |
| 5. CoT 변형 | `<think>...</think>` 같은 reasoning chain 학습 |
| 6. Multi-task mix | Instruction tuning에서 100여 dataset 섞는 것 |

→ **새로 배워야 할 robotics-specific 지식은 거의 없다**. 익숙한 도구들의 조합. RT-2가 robotics 진입의 좋은 첫 케이스인 이유.

---

## 부록: 인용 / 추가 자료

### A. 함께 읽기

- **[[OpenVLA]]** — RT-2의 오픈 버전. 다음 정독 대상. 같은 token-based action, 오픈 weights/code.
- **[[RT-1]]** — 정독 대상은 아니지만 256 bin scheme + robot data 원본. Brohan et al. 2022, arXiv:2212.06817.
- **[[Diffusion-Policy]]** — RT-2와 대조되는 continuous action 접근. Phase 2 batch에서 수집.
- **[[ACT]]** — Action chunking의 시초. RT-2는 1-step만 예측하므로 chunking 도입 시 어떤 변화?

### B. 공식 자료

- Project page (videos, demos): https://robotics-transformer2.github.io
- Code/weight: **closed** (Google DeepMind 내부)
- Blog post: Google DeepMind 공식 announcement (있음, 별도 수집 권장)

### C. 본 요약 작성 중 발견한 참고자료

- **PaLI-X 원논문** (Chen 2023a, arXiv:2305.18565): RT-2-PaLI-X의 backbone 이해에 도움
- **PaLM-E 원논문** (Driess 2023, arXiv:2303.03378): backbone + earlier robotics VLM
- **Symbol tuning 원논문** (Wei 2023, arXiv:2305.08298): action token overwrite의 이론적 정당화

위 3개는 backbone 이해를 위해 참고할 가치 있음. Track A 보조 자료로 추가 권장 (다음 batch download 시 고려).

### D. 한계의 정량적 hint

논문의 Appendix G에서 명시한 failure case는 약 5개 카테고리. Section 4의 결과 표에서 Unseen Environment (Hard) 33~35%가 가장 약한 부분. 이게 sim-to-real, lighting, clutter generalization의 difficulty floor를 보여줌. **이후 모델들이 이 floor를 어떻게 끌어올리는가**가 후속 정독의 평가 축.
