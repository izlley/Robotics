---
name: pi0-pre-post-training
description: π0의 핵심 training recipe — diverse mixed-quality pre-train + high-quality task-specific post-train (LLM-style). Pre만이나 Post만은 brittle
metadata:
  type: learning
  created_at: 2026-05-23
  updated_at: 2026-05-23
  source: papers/core-models/π 0 - A Vision-Language-Action Flow Model for General Robot Control.pdf
---

# 한 줄 사실

π0는 LLM-style **pre-training + post-training 명시적 분리**를 robotics에 최초로 적용:
- **Pre-training**: 다양한 robot · 다양한 task · **mixed quality** data (전체 10K hours)
- **Post-training**: 단일 task · **high quality**만 (5-100 hours)

→ Pre-only는 zero-shot OK but dexterous task brittle. Post-only는 clean but mistake recovery 못 함. **둘 결합이 결정적**.

# 왜 이게 robotics에서 중요한가

**LLM에서는 이미 표준이지만, VLA에서는 OpenVLA 같은 모델이 단일 단계**:
- OpenVLA: 970K episodes를 한꺼번에 학습 (mixed quality)
- π0: 명시적 두 단계

이 차이의 효과를 π0 실험이 입증:

| 학습 condition | 5-20분 dexterous task 결과 |
|---|---|
| Pre-only (zero-shot) | 일부 task 가능 but 안정성 부족 |
| Post-only (scratch + task data만) | 깔끔하지만 mistake recovery 불가 |
| **Pre + Post (full)** | **>50% max score across all tasks** ★ |

# Mechanism — 왜 두 단계가 필요한가

## Pre-training 데이터의 역할

다양한 robot에서 수집한 mixed-quality data에는 **실수 + recovery example**이 자연스럽게 포함:
- Teleoperator가 가끔 잘못 잡음 → 다시 시도
- Object slip 후 re-grasp
- Misaligned approach → correction

이런 example은 **high-quality post-training data에는 거의 없음** (teleoperator가 한 번에 잘 수행한 episode만 선택됨).

→ Pre-training이 **distribution shift / mistake recovery에 robust**한 base를 만듦.

## Post-training 데이터의 역할

특정 task에 대해 **fluent, consistent, efficient**한 sample만 모음.
- 사용자가 원하는 quality와 style을 학습
- "어떻게 하면 잘 하는지"를 알려줌

→ Post-training이 **task 수행의 fluency**를 만듦.

## 둘이 결합하면

학습된 모델은:
- **Default 행동**: post-training의 fluent style
- **Recovery 행동**: pre-training의 robust mistake handling

이는 LLM의 base model + SFT/RLHF와 정확히 같은 분업.

# 데이터 사이즈 차이

| 단계 | 데이터 |
|---|---|
| Pre-training | 10,000 hours (903M timesteps own + OXE 9.1%) |
| Post-training | task당 **5-100 hours** |

→ Pre가 100~2000배 더 큼. 그러나 둘 다 필수.

# 응용 포인트

**우리 스터디에서 SmolVLA fine-tune 시 적용 가능**:
- SmolVLA 공식 weights = community data로 pre-trained
- 우리 새 task에 fine-tune할 때 high-quality demo만 모으면 post-training만 한 것
- **만약 fine-tune data에 mistake recovery example이 부족하면 robust성 결여 예상**

**우리 스터디에서 from-scratch 학습 시 적용**:
- 두 단계로 명시적 분리 권장
- Pre-training data를 의도적으로 다양화 + lower quality 허용
- Post-training data는 단일 task, high quality

**Track c new idea 후보**:
- Pre/post ratio sweep (pre 10K hours, post 5h / 20h / 100h)
- Pre-training mixture composition (어떤 task가 어떤 generalization에 도움?)
- "Synthetic mistake injection" — post-only data에 인위적 mistake/recovery 추가
- Curriculum (pre → post → pre → post)

# 한계·미해결

- Pre/post 분리의 ratio 최적값 미정 (논문도 "5-100 hours" 범위로만 명시)
- Pre-training composition의 best practice 미정 (저자도 future work로 인정)
- Post-training의 catastrophic forgetting 분석 없음 (pre의 다양성을 post가 망가뜨리는가?)
- 데이터 mixing 정량 분석 부족

# LLM 도구와의 관계

| LLM 단계 | π0 대응 |
|---|---|
| **Base model (pre-train)** | π0 pre-training (10K hours mixed) |
| **SFT (Instruction-tuned)** | π0 post-training (high-quality task data) |
| **RLHF / DPO** | π★0.6 의 RECAP RL (다음 정독에서 다룸) |
| **Continual learning** | 아직 robotics에서 표준화 안 됨 |

# 관련 메모리

- [[pi0-moe-single-transformer]]
- [[pi0-blockwise-causal-attention]]
- [[rt2-co-fine-tuning-trick]] — RT-2의 co-fine-tuning (다른 접근, 같은 정신)
- [[openvla-lora-recipe]] — OpenVLA fine-tuning (단일 단계 접근)
