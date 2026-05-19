---
name: openvla-quantile-discretization
description: Action discretization 시 min-max 대신 1-99 percentile을 쓰면 outlier에 robust, effective granularity ~100x 향상
metadata:
  type: learning
  created_at: 2026-05-19
  updated_at: 2026-05-19
  source: papers/core-models/OpenVLA-An-Open-Source-Vision-Language-Action-Model.pdf
---

# 한 줄 사실

OpenVLA는 RT-2의 256 bin action discretization을 **min-max → 1-99 percentile**로 바꿔서 outlier에 robust, 정상 동작의 정밀도를 ~100x 향상.

# Why (RT-2의 문제점)

RT-2 (Brohan 2023)는 각 action 차원을 다음으로 discretize:

$$
\text{bin}(a) = \left\lfloor \frac{a - a_{\min}}{a_{\max} - a_{\min}} \times 256 \right\rfloor
$$

문제: Robot trajectory data에는 outlier가 흔하다.
- 예: gripper 명령이 일반적으로 [0, 1] 범위인데, 가끔 +100 같은 spurious 값이 들어옴 (sensor noise, teleoperator 실수, frame drop 등)
- → $a_{\max} = 100$이 되면 정상 범위 [0, 1]이 전체 bin의 1% 안에 압축됨
- → effective granularity가 256 bin → 사실상 ~2-3 bin

# OpenVLA 해결책

$$
\text{bin}(a) = \left\lfloor \frac{a - q_{1\%}}{q_{99\%} - q_{1\%}} \times 256 \right\rfloor
$$

여기서 $q_{1\%}, q_{99\%}$는 training data 분포에서 계산한 1st, 99th percentile.

**효과**: 상위·하위 1% outlier 제거 → 정상 분포가 256 bin을 다 활용 → **effective granularity ~100x 향상**.

# 비자명한 이유

- 사소해 보이는 디테일이지만 **실제 manipulation 정밀도에 매우 큰 영향**
- 특히 OpenX-Embodiment 같은 multi-source data에서는 각 데이터셋의 collection error 분포가 달라 outlier가 더 많음
- min-max 정의는 LLM domain에서는 자연스럽지만(token id가 outlier 없음), robot domain에는 부적합

# 응용 포인트

**모든 후속 VLA 학습에서 quantile-based discretization을 default로 사용**:
- 우리 스터디에서 from-scratch VLA 만들 때
- SmolVLA, π0 등 후속 모델도 동일 또는 더 정교한 방식 사용 (정독 시 확인)
- 다른 percentile 값 (0.5-99.5, 5-95) sweep해서 sweet spot 정량화 가능 (Track c new idea)

# 한계·미해결

- 256 bin이 정말 충분한가? (dexterous task에서는 부족할 가능성 — π0의 continuous action 선택 이유)
- Bin 수 자체 sweep (128, 256, 512, 1024)도 가치 있음
- Percentile은 차원별로 독립 계산이 맞는가, joint distribution을 봐야 하나?

# 관련 메모리

- [[rt2-action-as-token-paradigm]] — RT-2의 원본 discretization scheme
- [[openvla-dual-vision-encoder-fusion]]
