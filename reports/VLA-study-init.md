# Robotics 분야 VLA 모델 트렌드 비교

> 기준일: 2026년 5월 18일  
> 주제: 로보틱스에서 사용되는 VLA, 즉 Vision-Language-Action 모델의 최근 트렌드와 주요 모델 비교

---

## 1. 요약

최근 로보틱스 VLA 모델은 단순히 이미지와 언어를 이해하는 수준을 넘어, 실제 로봇의 행동을 생성하는 **robot foundation model** 방향으로 발전하고 있다.

초기에는 VLM, 즉 Vision-Language Model에 action token을 붙여 로봇 행동을 예측하는 방식이 중심이었다. 그러나 최근에는 다음과 같은 방향으로 빠르게 이동하고 있다.

- **이산 action token → 연속 제어 action expert**
- **단일 로봇 학습 → cross-embodiment 학습**
- **robot arm 중심 → humanoid 중심**
- **cloud inference → on-device / lightweight inference**
- **imitation learning → RL 기반 self-improvement**
- **vision-language-action → tactile/force까지 포함한 VLA+**
- **단기 조작 → memory/world model 기반 long-horizon task**

현재 VLA의 핵심은 다음과 같이 정리할 수 있다.

> VLM 또는 LLM이 고수준 의미 이해와 명령 해석을 담당하고, 별도의 action expert, diffusion head, flow-matching head, controller가 실제 로봇 동작을 생성하는 구조로 진화하고 있다.

---

## 2. VLA란 무엇인가?

**VLA, Vision-Language-Action 모델**은 다음 세 가지 입력과 출력을 통합하는 로보틱스 모델이다.

| 구성 요소 | 의미 |
|---|---|
| Vision | 카메라 이미지, depth, multi-view observation |
| Language | 자연어 명령, task description, goal |
| Action | 로봇 joint command, end-effector pose, gripper command, action chunk |

즉, VLA는 다음과 같은 문제를 푸는 모델이다.

```text
이미지 + 자연어 명령 + 로봇 상태
→ 다음 로봇 행동
```

예를 들면 다음과 같다.

```text
Input:
- 카메라 이미지
- "빨간 컵을 집어서 바구니에 넣어"
- 현재 로봇 arm state

Output:
- end-effector 이동 경로
- gripper open/close command
- 다음 action chunk
```

---

## 3. VLA 아키텍처 변화

### 3.1 초기 구조: VLM + Action Token

초기 VLA는 이미지와 언어를 처리하는 VLM에 로봇 action을 token처럼 붙이는 방식이 많았다.

대표 모델:

- RT-2
- OpenVLA
- 일부 autoregressive VLA

특징은 다음과 같다.

```text
Image + Text
→ VLM
→ Action Token
→ Robot Command
```

장점:

- 기존 VLM의 semantic understanding 활용 가능
- 자연어 명령 처리에 강함
- unseen object나 language generalization에 유리

한계:

- 연속 제어에 부자연스러움
- 고주파 제어가 어려움
- 정밀 조작, 접촉 조작, deformable object 조작에 약함

---

### 3.2 최근 구조: VLM Backbone + Action Expert

최근 모델은 VLM이 모든 것을 직접 출력하지 않고, 별도의 action module을 둔다.

```text
Image + Text + Robot State
→ VLM / Multimodal Encoder
→ Action Expert
→ Continuous Action Chunk
→ Low-level Controller
```

대표 모델:

- π0
- π0-FAST
- SmolVLA
- GR00T N1
- Gemini Robotics
- Rho-alpha

장점:

- 연속 action 생성에 유리
- action chunking 가능
- 고주파 제어와 결합하기 좋음
- diffusion policy, flow matching, controller와 통합 가능

---

### 3.3 차세대 구조: VLA + Memory + World Model + RL

2025년 이후에는 VLA가 단순한 imitation learning 모델을 넘어 다음 요소와 결합하고 있다.

- Long-term memory
- Short-term embodied memory
- World model
- Video prediction
- Reinforcement learning
- Failure recovery
- Tactile/force feedback

차세대 구조는 다음과 비슷하다.

```text
Vision + Language + Robot State + Memory
→ Reasoning / Planning Module
→ World Model / Future Prediction
→ VLA Policy
→ Action Expert
→ Safety Filter
→ Robot Controller
```

---

## 4. 주요 VLA 모델 비교표

| 모델 | 공개 시기 | 개발 주체 | 핵심 방향 | Action 방식 | 공개성 | 주요 특징 |
|---|---:|---|---|---|---|---|
| RT-2 | 2023 | Google DeepMind | VLA 패러다임 정립 | Action token | Closed | 웹 VLM 지식을 로봇 제어로 전이 |
| RT-X / Open X-Embodiment | 2023 | Google DeepMind + 연구기관 | Multi-robot data scaling | RT 계열 | 부분 공개 | 22개 로봇 타입 데이터 기반 |
| Octo | 2024 | UC Berkeley / Stanford 등 | Generalist robot policy | Diffusion policy | Open | 오픈 generalist robot policy |
| OpenVLA | 2024 | Stanford / UC Berkeley 등 | 오픈소스 VLA | Action token | Open | 7B급 공개 VLA |
| π0 | 2024 | Physical Intelligence | General robot control | Flow/action expert | 부분 공개 | 고성능 generalist VLA |
| π0.5 | 2025 | Physical Intelligence | Open-world generalization | Flow/action expert | 부분 공개 | 다양한 로봇·웹 데이터 co-training |
| π0-FAST | 2025 | Physical Intelligence | Efficient action tokenization | FAST token | 부분 공개 | action sequence 압축 및 학습 효율화 |
| Gemini Robotics 1.5 | 2025 | Google DeepMind | VLA + embodied reasoning | Motion transfer 포함 | Closed | Gemini 기반 robotics model |
| GR00T N1 | 2025 | NVIDIA | Humanoid foundation model | VLA + diffusion/action stack | Open/customizable 주장 | humanoid 특화 |
| Helix | 2025 | Figure AI | Humanoid VLA | Proprietary | Closed | Figure humanoid 전용 |
| SmolVLA | 2025 | Hugging Face / LeRobot | 경량 오픈 VLA | Action chunk expert | Open | 450M급 경량 모델 |
| X-VLA | 2025/2026 | Tsinghua / Hugging Face | Cross-embodiment | Soft prompt + Transformer | Open | robot별 soft prompt 활용 |
| Rho-alpha | 2026 | Microsoft Research | VLA+ tactile/bimanual | Vision-language + tactile | Early access | 촉각·힘 feedback 결합 |

---

## 5. 모델별 상세 분석

---

## 5.1 RT-2

### 개요

RT-2는 Google DeepMind가 발표한 초기 대표 VLA 모델이다.  
기존 vision-language model을 로봇 데이터와 함께 학습하여, 이미지와 언어 입력을 로봇 action으로 변환한다.

### 핵심 아이디어

```text
Web-scale VLM knowledge
+ Robot trajectory data
→ Vision-Language-Action model
```

RT-2의 중요한 점은 로봇 action을 언어 token처럼 다룬다는 것이다.

### 장점

- VLA 패러다임을 정립함
- 웹 지식을 로봇 제어로 전이 가능
- 본 적 없는 물체나 개념에 대한 semantic generalization 가능성 제시
- 이후 OpenVLA, Gemini Robotics 등에 큰 영향

### 한계

- Closed model
- action discretization의 한계
- 고주파 연속 제어에는 부적합
- 실제 long-horizon task에는 제한적

---

## 5.2 Open X-Embodiment / RT-X

### 개요

Open X-Embodiment는 여러 연구기관이 참여해 만든 대규모 multi-robot dataset이다.

### 핵심 의미

로보틱스에서 가장 큰 병목은 데이터다.  
텍스트나 이미지와 달리 로봇 데이터는 수집 비용이 높고, 로봇마다 action space가 다르다.

Open X-Embodiment는 이 문제를 해결하기 위해 여러 로봇의 데이터를 통합했다.

### 중요성

- Cross-embodiment 학습의 기반
- 여러 로봇 타입 간 skill transfer 연구 촉진
- OpenVLA, Octo, π0, X-VLA 등 후속 모델에 영향

### 장점

- 다양한 로봇 데이터 통합
- multi-task, multi-robot 학습 가능
- robot foundation model의 데이터 기반 제공

### 한계

- 데이터 heterogeneity가 큼
- camera view, action space, gripper type 차이가 큼
- 데이터 표준화가 여전히 어려움

---

## 5.3 OpenVLA

### 개요

OpenVLA는 대표적인 오픈소스 7B급 VLA 모델이다.  
기존 closed VLA와 달리 모델과 코드, fine-tuning workflow가 공개되어 연구 커뮤니티에서 많이 활용된다.

### 구조

```text
Image + Language
→ VLM Backbone
→ Action Token Prediction
→ Robot Action
```

### 장점

- 공개 weight와 code 제공
- VLA 연구 baseline으로 적합
- fine-tuning 가능
- language grounding이 강함
- object variation이 있는 manipulation task에 유리

### 한계

- 7B급이라 추론 비용이 큼
- onboard inference에는 부담
- action tokenization 기반이라 정밀 연속 제어에 한계
- 실제 환경에서는 camera pose, lighting, latency에 민감

### 적합한 용도

- 연구실 VLA baseline
- robot manipulation fine-tuning
- language-conditioned manipulation
- LoRA / QLoRA 기반 adaptation 연구

---

## 5.4 π0 계열

### 개요

π0는 Physical Intelligence가 발표한 general robot control용 VLA 모델이다.  
최근 VLA 트렌드에서 가장 중요한 계열 중 하나다.

### 핵심 구조

π0는 VLM backbone에 별도의 action expert를 붙인다.

```text
Vision + Language
→ VLM Backbone
→ Action Expert
→ Continuous Robot Action
```

### π0 계열 발전

| 모델 | 특징 |
|---|---|
| π0 | General robot control용 VLA flow model |
| π0-FAST | FAST action tokenization 적용 |
| π0.5 | Open-world generalization 강화 |
| π0.6 / π\*0.6 | RL 기반 self-improvement |
| π0.7 | Steerable model, compositional generalization 강조 |

### 장점

- flow/action expert 기반이라 연속 제어에 강함
- laundry folding, table bussing, box assembly 등 복잡 task 데모
- open-world generalization 방향 제시
- RL과 결합해 배포 후 성능 개선 가능성 제시
- long-horizon manipulation에 강한 흐름

### 한계

- 완전 공개되지 않은 요소가 있음
- 데이터와 평가 재현성 제한
- 실제 산업 환경의 robustness는 별도 검증 필요
- safety layer와 controller 통합이 필수

---

## 5.5 Gemini Robotics 1.5

### 개요

Gemini Robotics는 Google DeepMind가 Gemini 기반으로 만든 robotics model이다.  
Gemini Robotics 1.5는 VLA와 embodied reasoning을 결합한 방향이다.

### 구조적 특징

```text
High-level reasoning
→ Embodied reasoning module
→ VLA execution model
→ Robot motor command
```

### 장점

- Gemini 계열의 강한 multimodal reasoning 활용
- multi-step task에 강함
- motion transfer 강조
- multi-embodiment robotics에 적합
- 고수준 planning과 저수준 action을 분리

### 한계

- Closed model
- 외부 연구자가 내부 구조와 데이터 검증 어려움
- 실제 deployment는 Google stack 의존 가능성
- latency와 safety 검증 필요

---

## 5.6 NVIDIA GR00T N1

### 개요

GR00T N1은 NVIDIA가 humanoid robot을 위해 발표한 robot foundation model이다.  
NVIDIA Isaac, Cosmos, simulation stack과 강하게 연결된다.

### 핵심 방향

```text
Humanoid robot
+ Simulation data
+ Synthetic data
+ VLA foundation model
→ Generalized humanoid skills
```

### 장점

- humanoid 특화
- NVIDIA Isaac Sim, Cosmos와 결합
- synthetic data generation 활용
- dual-arm, whole-body control 연구에 적합
- 산업용 humanoid 개발 workflow에 유리

### 한계

- NVIDIA 생태계 의존성
- humanoid control은 VLA만으로 해결 불가
- balance, collision avoidance, safety controller 필요
- 실제 hardware transfer가 어려움

---

## 5.7 Figure Helix

### 개요

Helix는 Figure AI가 자사 humanoid robot을 위해 만든 generalist humanoid VLA다.

### 특징

- Figure humanoid에 tightly integrated
- household object manipulation 강조
- 자연어 명령 기반 object handling
- 제품형 humanoid AI stack에 가까움

### 장점

- hardware와 AI가 통합되어 데모 완성도가 높음
- humanoid application 중심
- 실제 제품화 관점에서 빠른 iteration 가능

### 한계

- Closed model
- Figure robot 종속
- benchmark 기반 비교 어려움
- 연구 재현성 제한

---

## 5.8 SmolVLA

### 개요

SmolVLA는 Hugging Face / LeRobot 생태계에서 공개한 경량 VLA 모델이다.  
450M급 compact model로, 접근성과 저비용 실험을 강조한다.

### 구조

```text
Multi-camera views
+ Robot state
+ Language instruction
→ VLM features
→ Action expert
→ Action chunk
```

### 장점

- 경량 모델
- 오픈소스
- consumer-grade hardware에서 실험 가능
- LeRobot community dataset 활용
- 교육용, 연구용, 스타트업 PoC에 적합

### 한계

- 대형 VLA 대비 범용성 제한 가능
- 고난도 dexterous manipulation에는 한계
- task-specific fine-tuning 의존도 높음
- 데이터 품질에 민감

### 적합한 용도

- 저비용 로봇 arm
- 교육용 로보틱스
- 빠른 PoC
- DIY robot
- LeRobot 기반 실험

---

## 5.9 X-VLA

### 개요

X-VLA는 soft prompt를 활용해 다양한 robot embodiment를 하나의 모델에서 다루려는 cross-embodiment VLA다.

### 핵심 아이디어

로봇마다 다른 구조를 별도 모델로 학습하지 않고, robot/domain-specific soft prompt를 사용한다.

```text
Robot-specific soft prompt
+ Shared VLA backbone
→ Cross-embodiment policy
```

### 장점

- 서로 다른 로봇, 센서, action space를 하나의 모델에서 처리
- parameter-efficient adaptation 가능
- 새 로봇에 전체 모델 재학습 없이 적응 가능성
- multi-robot dataset scaling에 유리

### 한계

- 아직 연구/벤치마크 중심
- 실제 현장의 extreme domain shift 검증 필요
- contact-rich task에서는 추가 controller 필요

---

## 5.10 Microsoft Rho-alpha

### 개요

Rho-alpha는 Microsoft Research가 발표한 physical AI / robotics model이다.  
Phi 계열 vision-language model에서 파생되었으며, VLA에 tactile/force sensing을 결합하는 방향을 제시한다.

### 핵심 방향

```text
Vision + Language + Tactile/Force
→ Bimanual robot control
```

### 장점

- 촉각과 힘 feedback을 VLA에 통합
- bimanual manipulation에 초점
- plugging, insertion, contact-rich task에 유리
- simulation data와 human correction loop 활용

### 한계

- 초기 연구 단계
- tactile sensor 표준화 부족
- sim-to-real force transfer 어려움
- hardware cost와 integration complexity가 큼

---

## 6. 핵심 기술 트렌드

---

## 6.1 Action Token에서 Continuous Action Expert로 이동

초기 VLA는 action을 discrete token으로 예측했다.

```text
Action = token
```

최근 VLA는 다음 방식으로 이동하고 있다.

```text
Action = continuous trajectory
Action = action chunk
Action = flow-matched trajectory
Action = diffusion-generated trajectory
```

대표 흐름:

- π0의 flow/action expert
- diffusion policy 기반 robot policy
- FAST action tokenization
- SmolVLA의 action chunk
- GR00T의 action stack

### 의미

정밀 조작에는 단일 action token보다 연속 trajectory가 중요하다.

예:

- cloth folding
- cable manipulation
- wiping
- insertion
- bimanual assembly
- deformable object manipulation

---

## 6.2 Cross-Embodiment Generalization

VLA의 가장 중요한 과제 중 하나는 서로 다른 로봇 간 일반화다.

### 문제

로봇마다 다음이 다르다.

- arm kinematics
- gripper type
- camera position
- action dimension
- control frequency
- proprioception format
- end-effector convention

### 해결 방향

| 접근 | 설명 |
|---|---|
| Data standardization | 여러 로봇 데이터를 공통 포맷으로 정리 |
| Motion transfer | 한 로봇의 동작을 다른 로봇에 맞게 변환 |
| Soft prompt | robot-specific embedding 사용 |
| Adapter / LoRA | 로봇별 작은 모듈만 fine-tuning |
| Simulation | 다양한 embodiment 데이터를 synthetic하게 생성 |

대표 모델:

- Open X-Embodiment
- RT-X
- Gemini Robotics 1.5
- X-VLA
- GR00T
- π0.5

---

## 6.3 Humanoid VLA의 부상

2025년 이후 VLA는 robot arm에서 humanoid로 확장되고 있다.

대표 모델:

- NVIDIA GR00T N1
- Figure Helix
- Gemini Robotics
- 일부 π0 계열
- Tesla Optimus 계열 비공개 stack

### Humanoid VLA가 어려운 이유

Humanoid는 단순 arm보다 훨씬 복잡하다.

필요한 요소:

- whole-body balance
- dual-arm coordination
- locomotion
- manipulation
- collision avoidance
- human safety
- force control
- real-time control

따라서 humanoid에서는 VLA가 직접 모든 motor command를 담당하기보다 다음과 같은 hybrid 구조가 현실적이다.

```text
VLA:
- task understanding
- object selection
- high-level action

Controller:
- balance
- trajectory tracking
- collision avoidance
- safety
```

---

## 6.4 경량화와 On-device VLA

대형 VLA는 성능이 좋지만 로봇 제어에는 latency가 문제다.

### Cloud VLA

장점:

- 큰 모델 사용 가능
- reasoning 강함
- 업데이트 쉬움

단점:

- latency
- network dependency
- privacy
- safety risk

### On-device VLA

장점:

- low latency
- offline operation
- privacy
- safety control에 유리

단점:

- 모델 크기 제한
- 성능 제한
- hardware cost

대표 흐름:

- SmolVLA
- Gemini Robotics On-Device
- distilled OpenVLA
- X-VLA compact models
- LoRA/quantization 기반 VLA

---

## 6.5 RL 기반 Self-Improvement

Imitation learning만으로는 실제 환경의 실패를 모두 처리하기 어렵다.

### 문제

로봇은 실제 환경에서 다음 문제를 만난다.

- object slip
- friction variation
- imperfect grasp
- occlusion
- delayed feedback
- hardware wear
- actuator backlash
- sensor noise

### 해결 방향

VLA를 초기 policy로 사용하고, 이후 real-world experience로 개선한다.

```text
Pretrained VLA
→ Real-world rollout
→ Success/failure data
→ Offline RL / Online RL
→ Improved VLA policy
```

대표 흐름:

- π0.6 / π\*0.6
- RECAP
- ViVa
- world-model assisted RL

---

## 6.6 Tactile / Force 기반 VLA+

Vision만으로는 contact-rich manipulation을 안정적으로 수행하기 어렵다.

### Vision-only VLA가 어려운 task

- plug insertion
- force-sensitive grasp
- wiping
- cable manipulation
- screwing
- button pressing
- deformable object manipulation

### VLA+ 방향

```text
Vision + Language + Action
→ Vision + Language + Tactile + Force + Action
```

대표 흐름:

- Microsoft Rho-alpha
- Tactile-VLA
- HapticVLA
- force-aware manipulation models

---

## 7. 목적별 모델 선택 가이드

---

## 7.1 연구실 / 논문 연구용

추천 모델:

| 목적 | 추천 |
|---|---|
| VLA baseline 연구 | OpenVLA |
| 저비용 실험 | SmolVLA |
| cross-embodiment 연구 | X-VLA |
| action expert / flow VLA 연구 | π0 / openpi 계열 |
| generalist policy 비교 | Octo |

### 추천 조합

```text
OpenVLA + LoRA fine-tuning
SmolVLA + LeRobot dataset
X-VLA + multi-robot dataset
π0-style action expert + custom manipulation task
```

---

## 7.2 스타트업 / PoC

추천 방향:

| 상황 | 추천 |
|---|---|
| 빠른 PoC | SmolVLA |
| 고성능 manipulation | OpenVLA 또는 π0 계열 fine-tuning |
| humanoid 개발 | GR00T / Isaac stack |
| bimanual task | Rho-alpha류 접근 |
| 저비용 robot arm | LeRobot + SmolVLA |

### 현실적 전략

```text
1. SmolVLA 또는 OpenVLA로 빠른 baseline 구축
2. task-specific data 수집
3. LoRA / QLoRA fine-tuning
4. low-level controller와 safety layer 결합
5. 실패 데이터를 모아 iterative improvement
```

---

## 7.3 산업용 / 제품화

제품화에서는 모델 성능보다 다음 요소가 더 중요하다.

- reliability
- safety
- latency
- recovery
- monitoring
- data pipeline
- fleet learning
- hardware integration

추천 구조:

```text
High-level planner:
- LLM / VLM

Robot policy:
- VLA

Low-level control:
- classical controller
- MPC
- impedance control
- force controller

Safety:
- collision checker
- force limit
- emergency stop
- human detection

Learning loop:
- failure logging
- offline retraining
- RL fine-tuning
```

---

## 8. 현재 VLA의 한계

---

## 8.1 데이터 부족

로봇 데이터는 수집이 비싸다.  
인터넷 텍스트나 이미지처럼 쉽게 수십억 샘플을 모을 수 없다.

문제:

- teleoperation 비용
- robot wear
- reset cost
- sensor calibration
- environment diversity
- human annotation cost

---

## 8.2 Benchmark와 현실 차이

시뮬레이션이나 benchmark에서 높은 성능을 보이더라도 실제 환경에서는 다르다.

현실 변수:

- 조명 변화
- 카메라 위치 변화
- 물체 마찰
- 센서 노이즈
- gripper 마모
- 예측하지 못한 clutter
- 사람 개입
- network latency

---

## 8.3 Long-horizon Task 취약성

짧은 task는 가능해도 긴 task는 어렵다.

예:

```text
쉬운 task:
"컵을 집어라"

어려운 task:
"부엌을 정리하고, 식탁을 닦고, 컵을 싱크대에 넣고, 쓰레기를 버려라"
```

long-horizon task에는 다음이 필요하다.

- memory
- planning
- progress estimation
- failure recovery
- re-planning
- tool use
- safety monitoring

---

## 8.4 Safety 문제

VLA의 hallucination은 물리적 사고로 이어질 수 있다.

따라서 VLA output은 바로 actuator에 연결하면 안 된다.

필수 구성:

```text
VLA output
→ safety filter
→ collision checker
→ force limit
→ controller
→ actuator
```

---

## 8.5 Compute와 Latency

대형 VLA는 느릴 수 있다.  
로봇 제어는 보통 실시간성이 중요하다.

해결 방법:

- action chunking
- asynchronous inference
- distilled VLA
- small VLA
- on-device inference
- high-level VLA + low-level controller 분리

---

## 9. 향후 전망

---

## 9.1 VLA + World Model

로봇이 행동하기 전에 미래를 예측하는 구조가 중요해질 것이다.

```text
Current observation
+ Candidate action
→ Predicted future video/state
→ Value estimation
→ Action selection
```

---

## 9.2 VLA + RL

Foundation VLA로 초기 행동을 만들고, 실제 deployment에서 RL로 개선하는 방식이 늘어날 것이다.

```text
Pretraining
→ Imitation learning
→ Offline RL
→ On-robot fine-tuning
→ Fleet learning
```

---

## 9.3 Cross-Embodiment Adapter 표준화

로봇마다 다른 action space를 처리하기 위해 다음 방식이 중요해질 것이다.

- soft prompt
- robot adapter
- embodiment token
- action projection layer
- robot-specific LoRA

---

## 9.4 Multimodal Sensor 확장

기존 VLA:

```text
Vision + Language + Action
```

향후 VLA:

```text
Vision + Language + Proprioception + Depth + Tactile + Force + Audio + Action
```

---

## 9.5 Open-source와 Closed-source의 양극화

향후 시장은 두 방향으로 나뉠 가능성이 높다.

### Open-source 진영

- OpenVLA
- SmolVLA
- X-VLA
- LeRobot
- Octo
- openpi 계열

강점:

- 연구 확산
- 저비용 실험
- 커뮤니티 기반 개선
- 투명성

### Closed-source / Product 진영

- Gemini Robotics
- Figure Helix
- 일부 Physical Intelligence 모델
- 상용 humanoid stack

강점:

- 대규모 데이터
- 고성능 데모
- hardware-software 통합
- 제품화 속도

---

## 10. 실무 관점 핵심 체크리스트

VLA 모델을 선택하거나 도입할 때는 모델 이름보다 다음 항목이 중요하다.

### 10.1 데이터

- 내 task에 맞는 demonstration data가 있는가?
- camera view가 학습 데이터와 유사한가?
- object distribution이 충분히 다양한가?
- 실패 데이터도 수집하고 있는가?

### 10.2 로봇 호환성

- action space가 맞는가?
- gripper command가 호환되는가?
- control frequency가 맞는가?
- proprioception format이 맞는가?
- calibration이 안정적인가?

### 10.3 추론 성능

- onboard에서 돌릴 수 있는가?
- cloud latency를 감당할 수 있는가?
- action chunk size가 적절한가?
- real-time control loop와 분리되어 있는가?

### 10.4 안전성

- collision avoidance가 있는가?
- force limit이 있는가?
- emergency stop이 있는가?
- VLA output 검증 layer가 있는가?
- human-in-the-loop 개입이 가능한가?

### 10.5 일반화

- 새 물체에 대응 가능한가?
- 새 조명 조건에 강한가?
- 새 배경에서도 동작하는가?
- 새 로봇으로 transfer 가능한가?
- long-horizon task에서 recovery 가능한가?

---

## 11. 최종 정리

현재 Robotics VLA 모델 트렌드는 다음과 같이 정리된다.

| 트렌드 | 설명 |
|---|---|
| VLM → VLA | vision-language model이 실제 action 생성으로 확장 |
| Token → Continuous | discrete action token에서 continuous action expert로 이동 |
| Single robot → Multi-robot | cross-embodiment generalization이 핵심 |
| Arm → Humanoid | humanoid foundation model이 별도 카테고리로 부상 |
| Imitation → RL | deployment 경험으로 self-improvement |
| Vision-only → VLA+ | tactile, force, proprioception 결합 |
| Cloud → On-device | low-latency, privacy, safety를 위한 경량화 |
| Short task → Long-horizon | memory, planning, world model 필요 |

---

## 12. 결론

VLA는 로보틱스에서 가장 빠르게 발전하는 분야 중 하나다.  
그러나 아직 하나의 foundation model이 모든 로봇과 모든 환경을 안전하게 제어하는 단계는 아니다.

현재 가장 현실적인 접근은 다음과 같다.

```text
VLA
+ Task-specific fine-tuning
+ Low-level controller
+ Safety layer
+ Failure logging
+ RL / iterative improvement
```

즉, VLA는 기존 로보틱스 stack을 대체하기보다는, perception과 language understanding, high-level action generation을 담당하는 강력한 상위 policy로 사용되고 있다.

실무적으로는 다음 조합이 가장 현실적이다.

```text
LLM/VLM planner
→ VLA policy
→ action expert
→ safety filter
→ classical controller
→ robot hardware
```

따라서 VLA 도입의 핵심은 모델 자체보다도 다음 네 가지다.

1. 좋은 robot data pipeline
2. 안정적인 low-level controller
3. 확실한 safety/recovery layer
4. 지속적인 fine-tuning 및 real-world feedback loop