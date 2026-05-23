---
name: extended-curriculum-for-rfm-research
description: 원 Track A 8편으로는 RFM R&D 책무의 50-60%만 cover. 옵션 3(균형) 채택 — Track A 완료 + World Model 모듈 + LAPA + Q-Transformer 추가 정독 후 Track B/C
metadata:
  type: decision
  created_at: 2026-05-20
  updated_at: 2026-05-20
---

# Background

사용자 회사 업무 책무 (Robot Foundation Model R&D):

1. 최신 VLA / World Model / RFM 모델 논문 및 기술 분석
2. 대규모 로봇 학습 데이터(Real/Sim/Web) 수집·구성 전략 연구
3. RFM 사전학습 및 사후학습(SFT, RL) 성능 개선
4. data mixing 전략 및 generalization 향상 연구
5. 신규 모델 아키텍처 및 학습 방법론 개발
6. 최신 연구를 내부 RFM에 통합 및 개선 방향 도출

이에 대해 원 Track A([[baseline-papers-scope]])의 cover 비율 분석:

| 책무 | 현재 8편 cover | 갭 |
|---|---|---|
| VLA 모델 분석 | ~80% | 거의 OK |
| **World Model 분석** | ~10% | **별개 분야, 큰 누락 ★** |
| 데이터 전략 (Real/Sim/Web) | ~25% | Web video pretrain, sim2real 부족 |
| 사전학습 + SFT | ~70% | OK |
| RL 사후학습 | ~25% | π★0.6 1편만, framework 부족 |
| Data mixing + generalization | ~35% | 단편적 |
| 신규 아키텍처 | ~50% | memory/hierarchical/latent action 부재 |
| 내부 RFM 통합 | ~20% | 이론에 한정 |

**전체 약 50-60%만 cover**.

# 결정: 옵션 3 (균형)

Track A(8편 완료) → **핵심 추가 모듈만 정독** → Track B(hands-on) → Track C(회사 업무에서 자연스럽게).

옵션 1(모든 모듈 정독 후 Track B, 4-5개월)이나 옵션 4(Track A만 + 회사 업무 ad-hoc, 가장 lean)보다 사용자의 R&D 직무 + 시간 효율 균형이 좋음.

# 추가 정독 모듈 (Track A 종료 직후 진행)

## Module 3 — World Model (필수 3편)

VLA = "관측 → action" 매핑. World Model = "관측 + action → 다음 관측" 예측. RFM의 별개 한 축이며, 회사 업무의 "World Model 분석"에 필수.

1. **Dreamer V3** (Hafner et al. 2024) — world model + planning 기본형. arXiv:2301.04104
2. **GAIA-1** 또는 **GAIA-2** (Wayve) — autonomous driving world model. 산업 응용
3. **NVIDIA Cosmos** (2025) — 대규모 video world model, robotics 응용

후순위 (ad-hoc):
- 1X World Model (humanoid 특화)
- Genie / Genie-2 (DeepMind, interactive world model)
- World Labs (Fei-Fei Li, 2025+)
- UniSim, IRASim

## Module 4 — 데이터 전략 (필수 2편)

3. **Open X-Embodiment** (Padalkar et al. 2023, arXiv:2310.08864) — 22+ embodiment 통합. 원래 보조자료 계획됨. 본 모듈에 정식 편입
4. **LAPA — Latent Action Pretraining** (Ye et al. 2024, NeurIPS) — **action 없는 web video를 VLA pretrain에 활용**. 사용자의 "Web 데이터 활용 전략"에 직접 대응 ★

후순위:
- DROID (Khazatsky 2024)
- Ego4D + action labeling 응용
- RoboGen (sim data 자동생성)
- Sim2Real survey

## Module 5 — RL 심화 (필수 1편)

5. **Q-Transformer** (Chebotar et al. 2023, CoRL) — Transformer 기반 offline RL for robotics. π★0.6보다 더 일반적 framework

후순위:
- SERL / HIL-SERL (sample-efficient real-world RL)
- DPO/PPO for VLA
- HumanoidVerse, BARN

## 보조 자료 (Track A에 interleave 또는 함께 정독)

원래 [[baseline-papers-scope]]에서 계획된 것:
- **Diffusion Policy** (Chi 2023, RSS) — flow matching의 사촌, π0 이해에 도움
- **ACT** (Zhao 2023, ALOHA) — action chunking 시초
- VLA 최신 survey 1편 (있을 때)

# 진행 순서 (확정)

```
Phase 3 (현재 진행):
  ✓ RT-2 (session-002)
  ✓ OpenVLA (session-003)
  ✓ SmolVLA (session-004)
  → π0 (다음)
  → π0.5
  → π★0.6
  → π0.7
  → GR00T N1
  [3/8 완료, 5편 잔여]

Phase 3.5 — 보조 자료 (Track A에 끼움):
  → Diffusion Policy (π0 정독 직전 또는 직후)
  → ACT (보조)
  → Open X-Embodiment (OpenX 데이터 이해)
  → 최신 VLA survey 1편

Phase 4 — 확장 모듈 (Track A 직후):
  → Dreamer V3
  → GAIA-1/2
  → NVIDIA Cosmos
  → LAPA
  → Q-Transformer

Phase 5 — Synthesis (이론 종합):
  → reports/synthesis/architecture-evolution.md
  → reports/synthesis/data-and-scaling.md
  → reports/synthesis/evaluation-protocols.md
  → reports/synthesis/world-model-vs-vla.md (옵션 3 추가)

Phase 5.5 — Pre-Track-B Infrastructure (2026-05-23 추가):
  → LeRobot paper (Cadene 2026, arXiv:2602.22818) ★
  → (옵션) openpi tech report, NVIDIA Isaac docs
  - System paper, 300-400줄 정도. 9-섹션 풀 템플릿 사용 안 함
  - 이유: Track B의 main stack 자체에 대한 이해 선행 → B1 환경 setup 비용 ↓
  - SmolVLA 저자팀의 라이브러리이므로 SmolVLA 정독과 연결성 높음

Track B — Hands-on:
  → 환경 setup (LeRobot)
  → SmolVLA / OpenVLA inference + LoRA reproduction
  → Small from-scratch pretrain (mixing ablation)
  → RL fine-tune (Q-Transformer / RECAP 재현)
  → Architecture ablation

Track C — New idea:
  → 회사 업무 속에서 자연스럽게 (Track A+B 후)
```

# 분량 추정

- Phase 3 잔여: 5편 × 1 turn = ~5 turns
- Phase 3.5 보조: 4편 × 1 turn = ~4 turns (또는 본 정독에 함께)
- Phase 4 확장: 5편 × 1 turn = ~5 turns
- Phase 5 Synthesis: 2-3 turns
- **Phase 5.5 Pre-Track-B (LeRobot paper + 옵션): 1-2 turns**
- **이론 총 ~17-19 turns**
- Track B (hands-on): 별도, ~10-20 sessions
- 전체 약 2-3개월 (사용자 페이스)

# 후순위 (현재 미반영, 필요 시 ad-hoc)

| 영역 | 왜 후순위 |
|---|---|
| Production engineering | 회사 업무에서 자연스럽게 학습 가능 |
| Hardware co-design | 필요 시점에 한정적 |
| 추가 RL 논문 (SERL, HumanoidVerse 등) | Q-Transformer 후 필요성 판단 |
| 추가 architecture (HAMSTER, X-VLA, FAST) | 신규 아키텍처 개발 시점에 ad-hoc |
| 추가 World Model (1X, Genie, World Labs) | Module 3 정독 후 흥미·필요에 따라 |
| Tactile / force VLA (Rho-alpha 등) | sensor 확장 시점 |

# 재검토 조건

- **World Model 분야에 새 frontier 등장 시** (예: World Labs 정식 paper) → Module 3 우선순위 조정
- **회사 업무 우선순위 변화** (특정 axis 집중 필요 시) → 해당 영역 정독 가속
- **Track B 진입 후 이론 부족 발견** → 후순위 자료 정독으로 복귀
- **3개월 후 시점에 재평가** — Phase 5 종료 시점에 진도·만족도 점검

# 영향

- **Phase 3에서 변화 없음**: π0 → π0.5 → π★0.6 → π0.7 → GR00T 순서 그대로
- **Phase 3.5/4 신설**: 보조 자료와 확장 모듈을 명시적 phase로 추가
- **Track B 확장**: 단순 inference smoke test가 아니라 from-scratch pretrain·mixing ablation 포함하는 방향
- **SESSION_HANDOFF 갱신 필요**: 로드맵 전체 표시

# 관련 메모리

- [[baseline-papers-scope]] — 원본 8편 결정 (본 메모리가 확장)
- [[track-selection]] — Track A 선택 근거
- [[summary-style]] — 추가 정독에도 같은 self-contained 학습 교재 스타일 적용
- [[light-code-checkpoints]] — 추가 정독 중에도 light 체크인 가능
- [[user-role]] — dnotitia LLM 엔지니어, RFM R&D 책무
