---
name: pi0-moe-single-transformer
description: π0의 핵심 — 단일 transformer 안에 VLM expert + Action expert 두 weight set이 self-attention layer에서만 상호작용
metadata:
  type: learning
  created_at: 2026-05-23
  updated_at: 2026-05-23
  source: papers/core-models/π 0 - A Vision-Language-Action Flow Model for General Robot Control.pdf
---

# 한 줄 사실

π0는 **단일 transformer** 구조에 두 set의 weight (VLM expert from PaliGemma + Action expert 300M)를 두고, 토큰 type에 따라 다른 expert로 routing. 두 expert는 **self-attention layer에서만 상호작용** (각자 자기 W_Q, W_K, W_V; FFN은 따로).

# 왜 비자명한가

본 프로젝트가 다룬 다른 모델들과 다른 architecture pattern:
- **OpenVLA**: 한 transformer, 한 weight set (모든 token에 동일 weight)
- **SmolVLA**: 두 separate transformer (VLM + action expert) + interleaved cross-attention
- **π0**: 한 transformer, 두 weight set (token type별 routing)

이는 **LLM Mixture-of-Experts(MoE)**의 변형이지만:
- **Switch Transformer/GLaM**: learnable gating + sparse activation
- **π0**: **deterministic routing (token type 고정) + dense activation**

# 메커니즘 세부

```python
# pseudocode (one transformer layer)
def layer(tokens, types):
    # Per-token: type별 W_Q, W_K, W_V 선택
    Q_all, K_all, V_all = [], [], []
    for token, ttype in zip(tokens, types):
        expert = vlm_expert if ttype in ['image','language'] else action_expert
        Q_all.append(expert.W_Q(token))
        K_all.append(expert.W_K(token))
        V_all.append(expert.W_V(token))
    
    # Self-attention: 모든 token Q가 모든 K, V와 상호작용 (블록 mask 적용)
    output = attention(Q_all, K_all, V_all, blockwise_causal_mask)
    
    # FFN: 다시 type별로
    final = []
    for tok, ttype in zip(output, types):
        expert = vlm_expert if ttype in ['image','language'] else action_expert
        final.append(expert.FFN(tok))
    return final
```

# Dimension 차별화

VLM expert와 action expert가 동일 layer 안에 있지만 **width 달라도 됨**:
- VLM expert: width = 2048, mlp_dim = 16384 (Gemma 2B 그대로)
- Action expert: width = 1024 (절반), mlp_dim = 4096 (¼) — inference 가속용

→ self-attention 시 두 expert의 features가 같은 attention space에서 만나려면 attention head dimension 정렬만 필요 (둘 다 head_dim = 256, num_heads = 18).

# 응용 포인트

- **OpenVLA / SmolVLA 비교 실험**: 같은 robot task에 3가지 architecture (one transformer / two separate transformers / MoE single transformer)로 학습하면 trade-off 정량화 가능
- **Action expert 다운사이징**: 1024 width는 임의 선택. Sweep 가능 (Track c)
- **Learnable routing**: π0는 fixed routing이지만 Switch Transformer-style learnable router로 확장 가능
- **Multi-action expert**: action token을 여러 expert(arm용, base용, gripper용)로 더 분기 가능

# 한계·미해결

- π0가 fix routing을 쓰는 이유 미설명 (learnable이 더 좋은가? 검증 없음)
- 두 expert dimension 차이가 self-attention 호환성에 미치는 영향 미분석
- 더 많은 expert (3개 이상)로의 확장 미시도

# LLM/VLM 도구와의 관계

| LLM 도구 | π0의 응용 |
|---|---|
| Switch Transformer (sparse MoE) | π0는 dense + fixed routing version |
| Mixture-of-Modality-Experts | 같은 정신: modality별 weight |
| Transfusion (Zhou 2024) | π0 직접 영감 — single transformer로 text + image 학습 |
| BERT의 segment embedding | type별 다른 표현 학습 — π0의 routing이 한 단계 더 |

# 관련 메모리

- [[smolvla-flow-matching-vs-token]] — SmolVLA의 다른 architecture choice (two separate transformers)
- [[pi0-blockwise-causal-attention]] — π0의 attention mask 디테일
- [[pi0-pre-post-training]] — π0의 학습 recipe
