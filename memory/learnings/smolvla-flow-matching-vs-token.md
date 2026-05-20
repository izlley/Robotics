---
name: smolvla-flow-matching-vs-token
description: VLA의 두 패러다임 — token-based (RT-2/OpenVLA, discrete 256 bin) vs flow-matching expert (SmolVLA/π0, continuous). 후자가 정밀 제어·multimodal action에 우월
metadata:
  type: learning
  created_at: 2026-05-20
  updated_at: 2026-05-20
  source: papers/core-models/SmolVLA-A-Vision-Language-Action-Model-for-Affordable-and-Efficient-Robotics.pdf
---

# 한 줄 사실

VLA에는 본질적으로 두 패러다임이 있다:
- **Token-based** (RT-2, OpenVLA): action을 256 bin discretize → LLM이 next-token으로 예측. 단순, LLM 인프라 재활용.
- **Flow-matching expert** (SmolVLA, π0): VLM은 perception만, **별도 transformer**(action expert)가 flow matching으로 continuous action 생성. 정밀·multimodal action에 우월.

SmolVLA(0.45B, flow-matching)가 OpenVLA(7B, token-based)를 LIBERO에서 76.5% → 87.3%로 능가한 사실이 두 패러다임의 trade-off를 보여줌.

# 두 패러다임 비교

| 차원 | Token-based | Flow-matching expert |
|---|---|---|
| Action 표현 | 256 bin × N차원 = N tokens | continuous vector (action chunk) |
| 생성 메커니즘 | Next-token prediction (NLL) | Flow matching (MSE on velocity field) |
| Loss | Cross-entropy on action tokens | $\| v_\theta(A^\tau, o) - (\epsilon - A) \|^2$ |
| Inference | 1 forward = 1 action (or 1 token at a time) | ODE 적분 10~50 step → action chunk |
| 정밀도 | 256 bin ceiling | Continuous |
| Multimodal action | Single mode (argmax) | Naturally multimodal (sampling) |
| 학습 효율 | LLM SFT 그대로 | Diffusion-style 학습 (느림) |
| Inference 속도 | 빠름 (single forward) | 약간 느림 (10 step) |
| VLM 활용 | Backbone 전체 fine-tune | **Backbone frozen 가능** ← 효율 ↑ |
| Action chunk 자연 지원 | No (1 step 예측) | Yes (chunk 단위 생성) |

# 왜 flow-matching이 우월한가 (mechanism)

**1. Multimodal action distribution**
- 같은 (image, instruction) 상황에서 valid한 action이 여러 개 존재 (예: 사과를 왼쪽으로 집을지 오른쪽으로 집을지)
- Token-based는 NLL로 학습 → 모든 demo의 평균으로 mode-seeking. Argmax 시 한 mode만 선택.
- Flow-matching은 noise → data로의 velocity field 학습 → sampling으로 여러 valid action 생성 가능.

**2. Continuous precision**
- 256 bin discretize → 차원당 최대 256개 valid value. Dexterous task에서 부족.
- Flow-matching은 continuous → 무한 정밀도. Cloth folding, insertion 같은 정밀 task에 유리.

**3. Smooth trajectory**
- Token-based는 각 step 독립적 → 인접 action이 jitter할 수 있음.
- Flow-matching은 chunk 전체를 같이 생성 → 자연스럽게 smooth.

# Trade-off (왜 token-based가 사라지지 않는가)

**Token-based의 장점**:
- LLM/VLM 인프라(HuggingFace, FSDP, LoRA) 그대로 재활용
- Symbolic reasoning, chain-of-thought 같은 LLM 도구와 결합 자연스러움 (RT-2의 CoT 예시)
- Web data co-training이 쉬움 (text token과 action token이 같은 vocab)
- 학습 수렴 빠름

**Flow-matching의 단점**:
- 별도 transformer 필요 (architecture 복잡도↑)
- 학습 더 느림 (Beta sampling, MSE 계산)
- CoT, symbolic reasoning 결합 어려움
- Web data co-training 자연스럽지 않음

# 정량 비교

| Model | Params | Paradigm | LIBERO Avg |
|---|---|---|---|
| OpenVLA | 7B | Token | 76.5 |
| **π0 (full)** | **3.3B** | **Flow-matching** | **86.0** |
| **SmolVLA** | **0.45B** | **Flow-matching** | **87.3** |
| π0 (PaliGemma 3B, no robotics PT) | 3B | Flow-matching | 71.8 |

→ 같은 robotics pretrain 조건에서 flow-matching이 **+10pp 이상**.

# 응용 포인트

**우리 스터디 다음 정독**:
- π0 (SmolVLA의 큰 형): flow-matching action expert의 원조 → 정독 핵심
- π0-FAST: 둘의 장점 결합 시도 (continuous + token 같은 tokenization)
- GR00T N1: flow matching 대신 diffusion 사용 (사촌 격)

**우리 스터디 Track c (new idea) 후보**:
- Token-based VLA(OpenVLA)에 flow-matching head 추가하여 직접 비교
- Flow-matching VLA(SmolVLA)에 CoT 추가 시도 (어려움)
- Web co-training이 flow-matching에 어떻게 가능한지 (action을 token화 안 하고)

# 한계·미해결

- 두 패러다임의 hybrid는 가능한가? (FAST tokenization이 시도)
- Multimodal action이 정말 필요한가? Imitation learning에서는 demonstrator 한 명의 mode만 따라가도 OK일 수 있음
- Flow-matching의 inference 속도가 token-based보다 항상 느린가? Few-step distillation 가능성

# 관련 메모리

- [[rt2-action-as-token-paradigm]] — Token-based의 원조
- [[smolvla-vlm-layer-skipping]] — Flow-matching paradigm에서 VLM 효율화
- [[smolvla-async-inference]] — 두 paradigm 모두에 적용 가능한 system-level optimization
