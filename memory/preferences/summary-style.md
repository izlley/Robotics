---
name: summary-style
description: 사용자는 논문 원문 대신 reports/papers/*.md를 통해 학습한다. 요약은 self-contained·상세·수식 포함이어야 한다.
metadata:
  type: preference
  created_at: 2026-05-18
  updated_at: 2026-05-18
---

# 핵심 규칙

사용자는 **논문 원문을 직접 읽지 않고** `reports/papers/<model>.md` 문서로 학습한다. 따라서 요약 문서는 **원문 대체용 학습 교재** 수준이어야 한다.

# Why (계기)

2026-05-18 session-002 (RT-2 정독) 직후 사용자가 명시적으로 표명:
> "저는 paper를 정독하기 보다는 당신이 정리한 md 문서를 가지고 학습할 것이니 주요 내용과 수식은 이해하기 쉽게 꼭 상세히 설명해주세요."

# How to apply

## 1. Self-contained 원칙

- 문서만 읽어도 논문의 핵심 80%를 이해할 수 있어야 함
- "이건 논문 참조" 같은 회피 표현 금지
- Figure 번호만 인용하지 말고 figure가 보여주는 내용을 글로 풀어 쓰기

## 2. 수식·notation 적극 포함

- 논문이 명시한 수식은 반드시 포함하고 각 기호를 한 줄씩 정의 (예: "p(y_t | y_{<t}, x): 이전 토큰들과 입력 x가 주어졌을 때 t번째 토큰의 확률")
- 논문에서 수식이 없이 글로만 설명된 핵심 메커니즘은 **직접 수식화하여 보충** (예: action discretization, co-fine-tuning loss mixture)
- LaTeX 인라인 `$ ... $` 또는 블록 `$$ ... $$` 사용. 또는 ASCII 표기로 풀어쓰기

## 3. 구체적 예시 필수

- 추상적 설명 뒤에는 즉시 **구체 수치 예시** 첨부
- 예: "256 bin discretize" → "Δpos_x ∈ [-0.05m, +0.05m]를 균등 256 bin으로 나누면 bin width = 0.39mm, bin index 128이면 Δpos_x ≈ 0"
- 입력/출력 sequence 실제 token string도 예시로

## 4. 이해를 돕는 비유 (analogy)

- 사용자가 익숙한 LLM/VLM 개념과의 1:1 대응 적극 활용
- 예: "symbol tuning은 LLM에서 `<eos>` 같은 special token 추가하는 것과 같다", "co-fine-tuning은 instruction tuning에서 base data를 함께 흘리는 것과 같다"
- 표 형태로 정리하면 한눈에 보임

## 5. 메커니즘 설명

- "X가 Y를 한다"만 쓰지 말고 **왜 그 메커니즘이 작동하는지** 까지 설명
- 예: "co-fine-tuning이 generalization을 향상시킨다" + "→ 왜냐하면 fine-tune 단계에서 web concept이 weight space에 계속 인입되어 robot data로 인한 catastrophic narrowing을 막기 때문"

## 6. 한계의 mechanism

- "X가 안 됨" → "왜 안 되는가 (구조적 원인)" → "후속 모델은 어떻게 해결하려 했는가"
- 한계 분석이 후속 논문 motivation으로 자연스럽게 이어지도록

## 7. 분량은 학습에 충분하면 됨

- 9-섹션 풀 요약 분량: **400~800 줄 권장** (현재 RT-2.md ~314 줄은 부족할 수 있음)
- 기초 기술 / survey는 더 짧아도 OK (별도 템플릿)
- 분량보다 *완결성*이 우선 — 빈 곳 없이, 의문 없이 읽힐 것

## 8. 시각화

- ASCII 다이어그램 적극 활용 (graph TB 같은 mermaid보다 호환성 ↑)
- 표는 풍부하게 — 비교, 수치, 매핑 모두 표로
- code block에 pseudocode 사용 가능

# 적용 범위

- **즉시 적용**: 이미 작성된 `reports/papers/RT-2.md`도 본 기준으로 revise
- **앞으로 모든 논문 요약**에 적용
- 기초 기술 논문은 풀 9-섹션은 아니어도 위 원칙(수식·예시·analogy·메커니즘) 동일 적용
- Synthesis 문서, external blog 정리도 동일 톤

# 관련 메모리

- [[user-role]] — LLM/VLM 배경이라 analogy 풍부히
- [[baseline-papers-scope]] — 8편 + 보조 자료 전체에 본 규칙 적용
