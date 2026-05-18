---
name: rt2-action-as-token-paradigm
description: RT-2의 "action을 text token으로 다룬다" 패러다임 — 이후 VLA 모델 흐름의 출발점
metadata:
  type: learning
  created_at: 2026-05-18
  updated_at: 2026-05-18
  source: papers/core-models/RT-2-Vision-Language-Action-Models-Transfer-Web-Knowledge-to-Robotic-Control.pdf
---

# 한 줄 사실

RT-2는 6-DoF robot action을 각 차원당 **256개 bin으로 discretize**한 뒤, VLM의 **기존 vocab 토큰을 빌려서 (또는 overwrite하여) action ID로 재해석**한다. 결과적으로 VLM 구조 변경·새 layer 추가 없이 robot 제어가 가능하다.

# 왜 비자명한가

LLM/VLM 엔지니어의 직관에서는 "continuous action = regression head"가 자연스럽다. RT-2는 그 반대: **action도 token이다**. 이게 작동하는 이유는:

1. **256 bin × 8 차원 = 9 토큰 sequence**로 충분히 표현 가능 (manipulation에 한해)
2. VLM의 tokenizer가 이미 정수 token을 갖고 있거나(PaLI-X), 안 쓰는 token을 overwrite할 수 있음(PaLM-E) → **"symbol tuning"** (Wei et al. 2023) 그대로
3. Next-token prediction loss가 그대로 behavior cloning loss가 됨 — pretraining objective와 통일

# 응용 포인트

- **OpenVLA**도 동일 패러다임 (Llama-2 7B + action token) — RT-2의 오픈 버전이라 보면 됨
- 반대 흐름은 **π0 / Diffusion Policy / ACT** — continuous action expert를 별도로 두는 방식. 두 패러다임의 trade-off가 향후 정독 핵심 비교축
- **Symbol tuning은 LLM 도구에서 그대로 가져온 것** — domain-specific token 추가하는 일반적 LLM 기법과 본질적으로 같음

# 한계 (왜 후속 모델이 다른 길로 가는가)

- Discretization → fine-grained continuous control 정밀도에 ceiling 있음
- High-frequency control (50+ Hz) 필요 시 token sequence 길이가 부담
- Dexterous manipulation, contact-rich task에서 token 표현이 부족
- → 후속 모델 (π0 등)이 flow matching / diffusion action expert로 이동한 핵심 이유

# 관련 메모리

- [[rt2-co-fine-tuning-trick]] — 같은 RT-2의 다른 핵심 발견
- [[baseline-papers-scope]]
