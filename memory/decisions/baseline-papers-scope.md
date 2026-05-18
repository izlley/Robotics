---
name: baseline-papers-scope
description: 정독 대상 핵심 8편 + 2nd tier/foundations/data/surveys 확장 자료 수집 정책
metadata:
  type: decision
  created_at: 2026-05-18
  updated_at: 2026-05-18
---

# 결정

Track A에서 다룰 자료를 다음 5개 카테고리로 정의하고, 정독 흐름과 interleave하여 수집한다.

## (1) 핵심 모델 — 풀 9-섹션 요약 (총 8편)

연대·난이도 순 정독 순서:

| # | 모델 | 출처 후보 | 역할 |
|---|---|---|---|
| 1 | RT-2 | arXiv 2307.15818 | VLA 패러다임 출발점, discrete action token |
| 2 | OpenVLA | arXiv 2406.09246 | 7B 오픈소스 표준, fine-tuning baseline |
| 3 | SmolVLA | arXiv 또는 HF tech report | 450M 경량 |
| 4 | π0 | 보유 PDF | flow matching action expert |
| 5 | π0.5 | 보유 PDF | open-world generalization |
| 6 | π*0.6 | 보유 PDF | RL self-improvement (RECAP) |
| 7 | π0.7 | 보유 PDF | steerable / compositional |
| 8 | GR00T N1 | NVIDIA tech report / arXiv 2503.14734 | humanoid foundation |

## (2) 2nd tier 모델 — 간략 요약

Octo, X-VLA, π0-FAST, Gemini Robotics 1.5 tech report

## (3) 기초 기술 — 핵심 아이디어·수식 중심

Diffusion Policy (Chi et al., 2023 RSS), ACT (Zhao et al., 2023 ALOHA), FAST tokenization (PI 2025)

## (4) 데이터·벤치마크 — 짧은 정리

Open X-Embodiment (Padalkar et al., 2023), DROID (Khazatsky et al., 2024), LIBERO (Liu et al., 2023), RoboCasa

## (5) Survey & 공식 blog — bullet 정리

VLA·robot foundation model survey 1~2편, Physical Intelligence 공식 blog (π0/0.5/*0.6/0.7), NVIDIA GR00T blog, Hugging Face SmolVLA blog, LeRobot 공식 docs

# 이유

- (1) 8편은 seed 문서 5장 비교표의 핵심 모델군과 정확히 일치. baseline부터 frontier까지 spectrum cover.
- (2)~(4)는 (1) 정독 시 자주 인용되는 prerequisite이므로 함께 봐야 깊이가 생김
- (5)는 closed model이거나 blog form으로만 공개된 자료를 흡수하기 위함

# 수집 흐름

3-batch interleave:
1. **1차 batch (Phase 2)**: RT-2 / OpenVLA / SmolVLA / GR00T N1 + Diffusion Policy + ACT + Open X-Embodiment + survey 1편
2. **2차 batch (π0 시리즈 정독 시점)**: π0-FAST, X-VLA, Octo, LeRobot docs
3. **3차 batch (synthesis 단계)**: Gemini Robotics, GR00T blog, DROID/LIBERO, 추가 발견 자료

# 디렉토리 분류

```
papers/
├── core-models/      # (1) 8편
├── related-models/   # (2) 2nd tier
├── foundations/      # (3) 기초 기술
├── data-benchmarks/  # (4) 데이터·벤치마크
└── surveys/          # (5) survey PDF
```

웹 자료(blog 등)는 markdown 캡처하여 `reports/external/<source>-<title>.md`.

# 영향

- 정독 분량: 핵심 8편(풀 9-섹션) + 2nd tier 4편 + 기초 3편 + 데이터 3~4편 + survey/blog ≈ 총 18~20개 자료
- 자료당 평균 1 turn 가정 시 약 20 turn (≈ 4~6주)
- 각 자료 요약은 `reports/papers/<model>.md`, `reports/external/<source>-<title>.md`에 보관

# 재검토 조건

- 1차 batch 정독 후 깊이·페이스 재평가 (사용자 피드백)
- 새로운 frontier 모델 등장 시 우선순위 재조정
- 사용자가 humanoid·tactile 등 특정 sub-topic을 우선화하길 원할 경우

# 관련 메모리

- [[track-selection]]
- [[persistent-memory-policy]]
