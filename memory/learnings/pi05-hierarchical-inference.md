---
name: pi05-hierarchical-inference
description: π0.5의 hierarchical inference — 같은 모델이 high-level subtask 예측 후 low-level action 생성. Chain-of-Thought의 robotics 응용
metadata:
  type: learning
  created_at: 2026-05-26
  updated_at: 2026-05-26
  source: papers/core-models/π0.5- a Vision-Language-Action Model with Open-World Generalization.pdf
---

# 한 줄 사실

π0.5는 단일 모델이 두 단계 추론 수행: (1) `"clean the kitchen"` → `"pick up the plate"` (high-level subtask text 예측), (2) subtask → action chunk (low-level flow matching). Chain-of-Thought의 robotics version. SayCan류 (분리된 VLM + policy)와 대조.

# 수식적 분해

전체 분포를 두 단계로:

$$
\pi_\theta(a_{t:t+H}, \hat{\ell} \mid o_t, \ell) = \pi_\theta(a_{t:t+H} \mid o_t, \hat{\ell}) \cdot \pi_\theta(\hat{\ell} \mid o_t, \ell)
$$

기호:
- $\ell$: high-level task ("clean the kitchen")
- $\hat{\ell}$: predicted subtask ("pick up the plate")
- $o_t$: observation
- $a_{t:t+H}$: action chunk

**중요한 가정**: action 분포가 $\ell$에 **직접 의존하지 않음** — $\hat{\ell}$만 본다. Clean separation.

# Inference loop

```python
while not done:
    o_t = observe()
    if time_to_replan_subtask:                # ~2s 주기 (subtask 길이)
        l_hat = autoregressive_decode(VLM, o_t, l)
        # ex: "pick up the plate"
    else:
        keep current l_hat
    A = flow_matching_decode(action_expert, o_t, l_hat)
    # 50 Hz 주기, 1초 분량 chunk
    execute(A)
```

두 추론이 **다른 frequency**:
- High-level: ~2s 주기 (subtask 단위)
- Low-level: 50 Hz (action chunk per second)

# 같은 모델 vs 분리 모델

| 접근 | 예시 | 장점 | 단점 |
|---|---|---|---|
| 분리 모델 (separate VLM + policy) | SayCan, OK-Robot, GPT-4 + RT-2 | VLM은 zero-shot OK | Policy가 VLM 명령 잘 못 따르면 책임 불명 |
| **같은 모델 (π0.5)** | π0.5 | End-to-end optimization, in-domain alignment | 모델 크기 부담 |

π0.5 ablation에서 직접 비교:

| Method | Performance |
|---|---|
| **π0.5 (own HL)** | best |
| Implicit HL (학습에 HL 포함, but 추론에 안 함) | close 2nd |
| Human HL (oracle) | 3rd |
| no-VI ablation | significantly down |
| no-WD ablation | significantly down |
| **GPT-4 HL** | **worst (in trained baselines)** |
| no HL (excluded everywhere) | minimal |

핵심 발견:
- **In-domain HL training이 GPT-4 zero-shot HL보다 좋음** — 일반 LLM의 robotics 적용 한계
- **Implicit HL이 explicit 거의 따라잡음** — training에 포함만 해도 효과
- **Verbal Instruction (VI) 결정적** — 11% data지만 제거 시 크게 떨어짐

# 왜 같은 모델이 두 단계 다 가능한가

같은 transformer의 다른 token type:
- Subtask 예측: text token autoregressive (FAST 학습으로 보존)
- Action 예측: action expert + flow matching

이는 π0.5의 hybrid (FAST + Flow) architecture 덕분 — text 능력과 action 능력 둘 다 보유.

# Chain-of-Thought / Test-time Compute 연결

LLM의 CoT:
- "Q: ... ?" → "Let me think step by step: ... Answer: ..."
- Reasoning text가 final answer 생성에 도움

π0.5의 hierarchical:
- (image, task) → "subtask label" → action
- Subtask가 reasoning step 역할 (intermediate semantic representation)

OpenAI o1의 test-time compute:
- 더 많은 thinking tokens = 더 좋은 answer
- π0.5: subtask explicit prediction = 더 좋은 action

→ **Chain-of-Thought이 robotics에 자연스럽게 transfer**. Embodied CoT (Zawalski 2024)와 정확히 같은 흐름.

# 응용 포인트

**우리 스터디 Track c new idea**:
1. **Subtask granularity sweep**: 0.5s / 2s / 5s / 10s 길이 subtask → optimal abstraction
2. **Multi-step CoT for robotics**: subtask 1개가 아니라 sequence (CoT planning)
3. **Subtask diversity 효과**: 다양한 표현 ("grasp X", "pick up X", "manipulate X") 학습 시
4. **LLM과 비교**: GPT-4, Claude, Llama-3 등 다양한 LLM as HL과 비교

**LLM 영역과의 mapping**:
- CoT (Wei 2022) → 같은 정신
- Tree of Thoughts (Yao 2023) → multi-subtask 가능성 탐색
- o1 / test-time compute (Jaech 2024) → subtask prediction을 reasoning tokens로
- Skill decomposition (Voyager 등) → robot skill hierarchy

# 한계·미해결

- Subtask가 **2초 단위로 fixed** — adaptive frequency 가능?
- High-level이 잘못된 subtask 선택하면 low-level 실패 (cascading error)
- Subtask label 자체의 vocabulary 한계 (학습 데이터의 표현으로 제약됨)
- Recovery from mistake: high-level은 잘못된 subtask 후 재시도 어려움 (causal mask)

# 관련 메모리

- [[pi0-pre-post-training]] — pre/post training 분리의 conceptual ancestor
- [[pi05-co-training-quantity-vs-diversity]] — VI data가 hierarchical에 결정적
- [[pi05-hybrid-fast-flow]] — text + action 둘 다 학습 가능하게 하는 architecture
- [[rt2-action-as-token-paradigm]] — RT-2 CoT가 conceptual seed
