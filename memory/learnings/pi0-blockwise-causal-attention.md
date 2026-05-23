---
name: pi0-blockwise-causal-attention
description: π0의 attention mask — 3 blocks (images+lang / state / action). 블록 내 bidirectional, 블록 사이 causal forward만
metadata:
  type: learning
  created_at: 2026-05-23
  updated_at: 2026-05-23
  source: papers/core-models/π 0 - A Vision-Language-Action Flow Model for General Robot Control.pdf
---

# 한 줄 사실

π0는 attention을 3 block으로 나눠 다음 패턴 적용:
- Block 1 (images + language): VLM pre-training input, 다른 block 못 봄
- Block 2 (robot state): Block 1만 봄, Block 3 못 봄 (KV cache 가능)
- Block 3 (action chunk): 모든 block 봄 + 블록 내 bidirectional

# 메커니즘

```
                Block1     Block2    Block3
              (img+lang)  (state)  (action)
              ┌─────────┬────────┬─────────┐
   Block1     │  ↔ ↔ ↔  │   ✗    │    ✗    │  ← 이전 blocks 못 봄
              ├─────────┼────────┼─────────┤
   Block2     │   ✓     │  (1)   │    ✗    │  ← 다음 block 못 봄
              ├─────────┼────────┼─────────┤
   Block3     │   ✓     │   ✓    │  ↔ ↔ ↔  │  ← 모든 거 보고 내부 bidirectional
              └─────────┴────────┴─────────┘
```

# 세 가지 design choice의 이유

## (1) Block 1이 future blocks 못 보는 이유

VLM pre-training (PaliGemma) 동안 이미지+텍스트만 봤음. 만약 state/action에 attend하게 두면:
- **Distribution shift**: 학습 시점 안 본 type의 token에 attend → VLM representations 망가짐
- **PaliGemma의 visual reasoning 능력 보존 실패**

→ Block 1은 자기 블록 안에서만 봐서 PaliGemma weights를 robot data로 fine-tune 시에도 visual reasoning 보존.

## (2) Block 2 (state)가 따로 block인 이유

State $q_t$는 한 inference call 안에서 **항상 동일** (action만 변화).
- **KV cache 활용**: state의 K, V를 한 번 계산 → ODE 10 step 모두에서 reuse
- 만약 Block 3(action)이 매번 다른 state를 본다고 가정하면 cache 불가
- Block 3에서 Block 2 attend는 OK (cache hit)

→ State KV cache가 **inference 시간의 큰 portion을 절약** (state forward pass = 0번 / 10 step).

## (3) Block 3 (action)이 bidirectional within인 이유

SmolVLA는 **causal within actions** (위치 t는 t-1까지만 attend).
π0는 **bidirectional within actions** (위치 t는 chunk 내 모든 다른 위치에 attend).

차이의 의미:
- Causal: autoregressive style. action 시퀀스가 시간순으로 생성되는 듯한 inductive bias.
- Bidirectional: chunk 전체가 동시 생성. 더 holistic.

Flow matching은 **chunk 전체를 noise→data로 변환**하므로 bidirectional이 자연스러움 (chunk가 한 번에 정해짐). Causal은 인공적 ordering.

# 응용 포인트

- **State KV cache**: π0의 73 ms 추론 시간 중 obs forward 32 ms (한 번만), action forward 10×2.7=27 ms. State cache 없으면 32 ms × 10 = 320 ms로 inference time이 ~5배 증가했을 것
- **Bidirectional within actions가 좋은가?** SmolVLA causal vs π0 bidirectional 비교 ablation 가능 (Track c)
- **Block 1 격리는 모든 VLA 학습에 적용 가능 원리** — VLM 능력 보존 보장

# LLM 도구와의 관계

| LLM 도구 | π0와의 관계 |
|---|---|
| KV cache (Transformer inference 최적화) | π0가 적극 활용 (state token) |
| Prefix LM mask (PaLM, T5) | π0의 block mask와 유사 |
| Bidirectional mask (BERT) | π0의 action block 내부에서 사용 |
| Causal mask (decoder LLM) | π0의 block 사이에서 사용 |
| Sliding window attention | π0와 무관 (chunk size 50으로 짧음) |

# 한계·미해결

- Block 분리가 항상 좋은가? 통합 mask와 비교 안 됨
- 4 block 이상으로 확장하면? (예: image, language, state, action을 모두 별도 block으로)
- Block 사이 attention sparsity가 perception flow 제약하는가?

# 관련 메모리

- [[pi0-moe-single-transformer]] — 같은 transformer의 weight side
- [[smolvla-flow-matching-vs-token]] — SmolVLA의 causal mask와 대조
