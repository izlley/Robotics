---
name: pi05-hybrid-fast-flow
description: π0.5의 hybrid action representation — Pre-training은 FAST discrete tokens, Post-training에 Flow matching expert 추가. FAST가 language ability 보존에 결정적
metadata:
  type: learning
  created_at: 2026-05-26
  updated_at: 2026-05-26
  source: papers/core-models/π0.5- a Vision-Language-Action Model with Open-World Generalization.pdf
---

# 한 줄 사실

π0.5는 **discrete action token (FAST)**과 **continuous flow matching** 표현을 모두 학습. Pre-training은 FAST로 빠르게, post-training에 flow matching expert 추가. 이 조합이 π0의 단점 (language ability 약함)을 해결.

# Two representations, two strengths

| 표현 | 장점 | 단점 |
|---|---|---|
| **FAST discrete** (autoregressive, cross-entropy) | 학습 빠름 (parallel), LLM 인프라 그대로, **language semantic 학습 well-suited** | 추론 시 autoregressive (slow), continuous precision 부족 |
| **Flow matching continuous** | 추론 빠름 (10 step ODE), continuous action, dexterous에 강함 | 학습 시 noise sampling overhead, language semantic 학습 약함 |

→ **둘 다 쓰자**: 학습 시 둘 다 supervise, 추론 시 task에 맞게 사용.

# 학습 단계별 동작

## Pre-training (280k steps)
- $\alpha = 0$ (cross-entropy only)
- 모든 action이 FAST encoded text tokens
- Action expert weights는 학습 안 됨 (random init 유지)
- Loss: $\mathcal{L}_{\text{pre}} = H(x, f^\ell_\theta)$ (cross-entropy)

## Post-training (80k steps)
- $\alpha = 10.0$ (joint loss)
- 두 표현 동시 학습:
  - Text tokens: cross-entropy (language + FAST action 모두)
  - Action expert: flow matching MSE
- Action expert가 random init부터 학습 시작
- Loss: $\mathcal{L}_{\text{post}} = H(x, f^\ell_\theta) + \alpha \cdot \|\omega - a - f^a_\theta(...)\|^2$

# 결정적 발견: FAST가 language ability를 결정

π0 (flow matching only) vs π0.5 (FAST + flow):

| Model | Language following |
|---|---|
| π0 (flow only) | low |
| π0-FAST+Flow (hybrid only, no HL/WD) | medium |
| **π0.5 (full, FAST pre-train + HL + WD)** | **high** |

해석:
- **Flow matching only pre-training = language semantics 학습 약함**
- Discrete cross-entropy가 **LLM-style language abilities를 보존**
- π0가 language following weak이 paradigm 자체 문제가 아님 → pre-training type 문제

# Attention masking으로 두 표현 격리

Figure 18의 attention mask는 정교:
- VLM 부분 (images + text + state): prefix bidirectional
- FAST tokens: autoregressive
- Flow expert tokens: bidirectional within chunk
- **FAST와 Flow expert가 서로 못 봄** ← 정보 leakage 방지

```
            VLM_prefix  FAST_tokens  Flow_expert
VLM_prefix:    bi-di        ^             ^
FAST:           v         causal          X
Flow expert:    v           X           bi-di
```

(^: future block 못 봄, X: blocked, v: previous block 봄, bi-di: bidirectional within)

# 왜 이게 작동하나 (mechanism)

1. **FAST pre-training**: 모델이 "이 image + state에서 action token 시퀀스는 무엇?" 학습
   - 동일한 mechanism으로 "이 image에서 caption은 무엇?" (WD)
   - "이 task에서 subtask는 무엇?" (HL)
   - → 모두 next-token prediction으로 통합. Multi-task synergy.

2. **Post-training flow expert 추가**: continuous precision이 필요한 dexterous action만 flow matching으로
   - FAST는 학습 그대로 유지 (language, subtask prediction에 사용)
   - Flow expert는 specialized for fast continuous control

3. **추론 시 분리**:
   - High-level subtask: FAST autoregressive decode (text)
   - Low-level action: Flow matching ODE (continuous)

# 응용 포인트

**우리 스터디에서 검증 가능**:
- 같은 모델을 (a) flow only / (b) FAST only / (c) hybrid로 학습 → language following 비교
- $\alpha$ sweep: 0.1, 1, 10, 100 → joint training balance
- FAST tokenizer 자체 효과 — FAST vs 단순 256 bin → FAST가 효율적인 이유

**LLM/VLM 분야와의 연결**:
- Hybrid AR + Diffusion (PixArt-α, Stable Diffusion 3에서 시도) → π0.5가 robotics에 적용
- Mixture of Experts (한 모델, 다른 representation) → π0.5의 mask로 격리
- Specialized expert (action expert만 따로) → modular design

# 한계·미해결

- FAST vs Flow 추론 시 어떤 task에 어느 게 좋은지 unclear (논문은 subtask=text, action=flow 고정)
- 두 표현이 서로 attention 못 보는 게 정말 optimal? — Cross-attention 허용 시 효과?
- α=10.0 magic number — sweep ablation 부재
- FAST의 compression rate (Pertsch 2025)가 다른 robot에서도 작동하나?

# 관련 메모리

- [[pi0-moe-single-transformer]] — 같은 MoE architecture, but action expert만 (FAST 없음)
- [[rt2-action-as-token-paradigm]] — token-based VLA의 원조 (FAST는 변형)
- [[smolvla-flow-matching-vs-token]] — token vs flow trade-off 분석
- [[pi05-co-training-quantity-vs-diversity]]
