# ROADMAP — VLA / RFM 학습 계획

> **Single source of truth.** 학습 계획에 대한 모든 정보는 이 파일에 통합됩니다. 매 session 종료 시 갱신.
> **Last updated**: 2026-05-23 (session-008, Phase 5.5 신설)
>
> 관련 문서:
> - [`CLAUDE.md`](CLAUDE.md) — 프로젝트 헌법 (변경 거의 없음)
> - [`SESSION_HANDOFF.md`](SESSION_HANDOFF.md) — 현재 작업 상태 (turn-by-turn 갱신)
> - [`memory/`](memory/) — 결정 히스토리 & 학습한 통찰
> - [`reports/papers/`](reports/papers/) — 정독한 논문 요약 (학습 교재)

---

## 0. 한 줄 정의

**VLA / World Model / RFM 분야 frontier 기술을 체계적으로 학습**하여 dnotitia 내부 RFM R&D에 활용하는 self-curriculum. 사용자 = LLM/VLM 엔지니어 정용엽 → 로보틱스 신규 진입.

## 1. 학습 절차 (큰 그림)

[CLAUDE.md §2](CLAUDE.md) 재서술:

| 단계 | 내용 | 현재 상태 |
|---|---|---|
| (a) | 기본 개념·이론 학습 | **🟢 진행 중** (Track A) |
| (b) | 오픈소스 활용 구현/학습/평가 실습 | ⏳ Track A 완료 후 |
| (c) | data/모델/학습/평가/추론 new idea 실험 | ⏳ Track C, 회사 업무 통합 |
| (d) | frontier 도달까지 반복 | 지속 |

## 2. Quick Status

- **진도**: Track A 핵심 8편 중 **3편 완료** (RT-2 ✅, OpenVLA ✅, SmolVLA ✅), **5편 잔여**
- **다음 turn**: π0 정독 (보유 PDF, 다운로드 불필요)
- **예상 종료**: 이론 ~15-20 turns (약 2-3개월 페이스), Hands-on 별도 ~10-20 sessions
- **확장 옵션**: [옵션 3 (균형 확장)](memory/decisions/extended-curriculum-for-rfm-research.md) 확정

## 3. 전체 로드맵

```
Phase 3 (현재): 핵심 8편 정독
   ↓
Phase 3.5: 보조 자료 (interleave 또는 직후)
   ↓
Phase 4: 확장 모듈 (World Model + LAPA + Q-Transformer)
   ↓
Phase 5: Synthesis (이론 종합 분석)
   ↓
Phase 5.5: Pre-Track-B Infrastructure Reading (LeRobot paper 등)  ← Track B 진입 직전
   ↓
Track B: Hands-on (LeRobot setup + reproduction + ablation)
   ↓
Track C: New idea & 회사 RFM 통합 (지속)
```

### Phase 3 — 핵심 모델 8편 정독

| # | 모델 | 출처 | Paradigm | 상태 | 산출물 |
|---|---|---|---|---|---|
| 1 | **RT-2** | Brohan 2023, Google DeepMind | Token-based | ✅ session-002 | [RT-2.md](reports/papers/RT-2.md) (~720줄) |
| 2 | **OpenVLA** | Kim 2024, Stanford/UCB | Token-based | ✅ session-003 | [OpenVLA.md](reports/papers/OpenVLA.md) (~820줄) |
| 3 | **SmolVLA** | HF 2025 | **Flow matching** | ✅ session-004 | [SmolVLA.md](reports/papers/SmolVLA.md) (~870줄) |
| 4 | **π0** | Black 2024, PI | Flow matching | 📍 **다음** | (보유 PDF) |
| 5 | **π0.5** | PI 2025 | Flow + open-world | ⏳ | (보유 PDF) |
| 6 | **π★0.6** | PI 2026 | Flow + RL (RECAP) | ⏳ | (보유 PDF) |
| 7 | **π0.7** | PI 2026 | Flow + steerable | ⏳ | (보유 PDF) |
| 8 | **GR00T N1** | NVIDIA 2025 | Diffusion + humanoid | ⏳ | 다운로드 필요 |

**Paradigm 분기**:
- **Token-based** (1, 2): action = text token 256 bin discretize
- **Flow matching expert** (3-7): VLM perception + 별도 flow matching transformer
- **Diffusion expert** (8): 사촌, action diffusion

배경: [memory/decisions/baseline-papers-scope.md](memory/decisions/baseline-papers-scope.md), [memory/decisions/track-selection.md](memory/decisions/track-selection.md)

### Phase 3.5 — 보조 자료 (Phase 3 도중 또는 직후 interleave)

| 자료 | 역할 | 정독 분량 | 우선순위 |
|---|---|---|---|
| **Diffusion Policy** (Chi 2023, RSS) | Flow matching의 사촌, π0 이해의 prerequisite | 풀 9-섹션 | High (π0 직전) |
| **ACT** (Zhao 2023, ALOHA) | Action chunking 시초 | 짧게 (핵심만) | Medium |
| **Open X-Embodiment** (Padalkar 2023) | OpenVLA·π0의 데이터 출처 | 짧게 (data perspective) | Medium |
| VLA 최신 survey 1편 (선정 TBD) | 큰 그림 갱신 | 짧게 | Low (Phase 4 직전) |

### Phase 4 — 확장 모듈 (Track A 종료 직후)

원 8편이 cover하지 못하는 axis 보강. 옵션 3 채택([memory/decisions/extended-curriculum-for-rfm-research.md](memory/decisions/extended-curriculum-for-rfm-research.md)).

#### Module 3 — World Model (필수 3편)

VLA = "관측 → action" vs World Model = "관측+action → 다음 관측". RFM의 별개 축.

| # | 논문 | 출처 | 역할 |
|---|---|---|---|
| 1 | **Dreamer V3** | Hafner 2024, arXiv:2301.04104 | World model + planning 기본형 |
| 2 | **GAIA-1** 또는 **GAIA-2** | Wayve | Autonomous driving 산업 응용 |
| 3 | **NVIDIA Cosmos** | 2025 | 대규모 video world model, robotics 응용 |

#### Module 4 — 데이터 전략 (필수 1편 신규)

| # | 논문 | 역할 |
|---|---|---|
| 1 | **LAPA — Latent Action Pretraining** (Ye 2024, NeurIPS) | **Web video → VLA pretrain 활용 method** ★ 사용자 데이터 전략 책무에 직결 |

(Open X-Embodiment는 Phase 3.5에서 다룸)

#### Module 5 — RL 심화 (필수 1편)

| # | 논문 | 역할 |
|---|---|---|
| 1 | **Q-Transformer** (Chebotar 2023, CoRL) | Transformer 기반 offline RL for robotics. π★0.6보다 더 일반적 framework |

### Phase 5 — Synthesis (이론 종합)

`reports/synthesis/` 하위 4편 작성:

| 문서 | 내용 |
|---|---|
| `architecture-evolution.md` | Token-based → flow matching → diffusion → world model 진화 흐름 |
| `data-and-scaling.md` | OpenX vs community vs web video. Mixing 전략 비교. |
| `evaluation-protocols.md` | LIBERO, MetaWorld, real-world benchmark 비교 |
| `world-model-vs-vla.md` | 옵션 3로 추가된 axis. 두 분야의 차이·교집합 |

이 4편은 향후 사용자가 회사에서 의사결정할 때 reference로 활용.

### Phase 5.5 — Pre-Track-B Infrastructure Reading (Track B 진입 직전)

이론(paradigm/model)이 아닌 **system/library/infrastructure** 자료를 정독하여 Track B 진입 비용 ↓.

| # | 자료 | 출처 | 역할 |
|---|---|---|---|
| 1 | **LeRobot paper** (Cadene et al.) | arXiv:2602.22818 (2026-02-26) | Track B의 메인 stack 자체. SmolVLA 저자팀이 만든 end-to-end robot learning library. Hardware ↔ Data ↔ Algorithm 통합 관점 ★ |
| 2 (옵션) | **openpi tech report** (Physical Intelligence) | 미정 | π0 시리즈의 reference 구현. 사용 결정 시 정독 |
| 3 (옵션) | **NVIDIA Isaac Sim / Lab docs** | NVIDIA 공식 | 시뮬레이션 환경. GR00T N1 hands-on 진입 시 |

**작성 규칙**:
- 9-섹션 풀 템플릿 ✗. **System paper용 별도 양식** (300-400줄, stack components / abstractions / API / hands-on relevance 중심).
- 또는 `reports/synthesis/` 안에 통합 정리 (Phase 5 synthesis 일부로)
- LLM 도구 analogy 포함 (HuggingFace transformers와의 비교 등)

**왜 이 시점인가**:
- Track B의 B1(환경 setup)이 부드러워짐 — LeRobot abstraction을 미리 익혀둠
- SmolVLA + LeRobot이 한 set으로 머릿속 묶임 (SmolVLA 정독에서 다룬 async inference, multi-camera 표준화 출처가 LeRobot)
- 너무 일찍 (예: Phase 3 중간) 읽으면 추상적이라 흐려짐

### Track B — Hands-on (이론 종료 후)

R&D 직무를 위한 full reproduction 수준 (단순 inference smoke test 아님):

| Session | 작업 |
|---|---|
| B1 | 환경 setup — LeRobot install, OpenVLA repo, simulators (LIBERO/RoboCasa) |
| B2 | SmolVLA / OpenVLA inference + LoRA fine-tune 재현 |
| B3 | 작은 from-scratch pretrain (data mixing ratio sweep) |
| B4 | RL fine-tune 재현 (Q-Transformer 또는 RECAP) |
| B5 | Architecture ablation (layer skipping N sweep, CA/SA 비율, flow vs diffusion head) |
| B6 | 결과 분석 + [memory/learnings/](memory/learnings/) 에 실전 발견 기록 |

추정: 10-20 sessions (이론 turn보다 길고 불확실).

**Light 코드 체크인**: 이론 중간에 5-10분 inference smoke test 끼울 수 있음 ([memory/preferences/light-code-checkpoints.md](memory/preferences/light-code-checkpoints.md)).

### Track C — New Idea & 회사 통합 (Track A+B 후 지속)

회사 업무 속에서 자연스럽게:
- 내부 RFM 개선 방향 도출
- 신규 아키텍처 제안 + 실험
- 데이터 mixing 전략 실험
- 최신 frontier 모델 분석 → 내부 통합

## 4. 분량 + 시간 추정

| Phase | 분량 | Turns | 누적 |
|---|---|---|---|
| Phase 3 잔여 (5편: π0, π0.5, π★0.6, π0.7, GR00T) | 5 paper | ~5 | ~5 |
| Phase 3.5 보조 (Diffusion Policy, ACT, OpenX, survey) | 3-4 paper (대부분 짧음) | ~3 | ~8 |
| Phase 4 확장 (Dreamer, GAIA, Cosmos, LAPA, Q-Transformer) | 5 paper | ~5 | ~13 |
| Phase 5 Synthesis (4편 통합 분석) | — | ~3 | ~16 |
| **Phase 5.5 Pre-Track-B (LeRobot paper + 옵션 자료)** | 1-2 paper (system, 짧음) | **~1-2** | **~17-18** |
| Track B Hands-on | reproduction + ablation | ~10-20 | ~27-38 |
| Track C | 회사 업무 지속 | — | — |

**이론 약 17-18 turns + hands-on 10-20 sessions. 약 2-3개월 (현재 페이스)**.

## 5. 사용자 회사 책무 vs 코스 cover

R&D 책무 6 axis에 대한 cover 비율:

| 책무 | 원 8편 (Phase 3만) | 옵션 3 완료 후 | 어느 phase가 채움 |
|---|---|---|---|
| VLA 모델 분석 | 80% | **95%** | Phase 3 + 3.5 |
| World Model 분석 | 10% | **70%** | **Module 3** ★ |
| 데이터 전략 (Real/Sim/Web) | 25% | **65%** | Module 4 + Phase 3.5 |
| 사전학습 + SFT | 70% | **85%** | Phase 3 + Track B |
| RL 사후학습 | 25% | **55%** | Module 5 + π★0.6 |
| Data mixing + generalization | 35% | **65%** | Synthesis + Track B |
| 신규 아키텍처 | 50% | **65%** | Phase 5 + Track B |
| 내부 RFM 통합 | 20% | **80%** | Track B + C |

→ **옵션 3 완료 시 평균 약 70-80% cover**. 나머지는 회사 업무 속 ad-hoc.

자세히: [memory/decisions/extended-curriculum-for-rfm-research.md](memory/decisions/extended-curriculum-for-rfm-research.md).

## 6. 후순위 (현재 미반영, 필요 시 ad-hoc)

| 카테고리 | 후순위 항목 | 언제 |
|---|---|---|
| World Model 추가 | 1X World Model, Genie, World Labs | Module 3 후 흥미·필요에 따라 |
| 데이터 추가 | DROID, RoboGen, Sim2Real survey | 데이터 axis 심화 필요 시 |
| RL 추가 | SERL, HumanoidVerse, DPO for VLA | Q-Transformer 후 |
| Architecture 추가 | HAMSTER, X-VLA, FAST, MoE-VLA | 신규 아키텍처 개발 시점 |
| Production engineering | Failure recovery, on-device | 회사 업무에서 자연스럽게 |
| Tactile/Force | Rho-alpha 등 | sensor 확장 시 |

## 7. 갱신 규칙

본 ROADMAP.md는 **각 session 종료 시 갱신**:
- Phase 3 한 편 정독 완료 → §3 표의 상태/링크 갱신
- 새 모듈 결정 → Phase 4 또는 §6 후순위에 추가
- 옵션 변경 → [memory/decisions/](memory/decisions/)에 새 결정 메모리 작성 + 본 파일 갱신
- 분량 추정 변화 → §4 표 갱신

**source of truth 원칙**:
- 본 파일 = "계획 (the plan)"
- [SESSION_HANDOFF.md](SESSION_HANDOFF.md) = "현재 상태 (current state)" — 짧게, 본 파일을 참조
- [memory/decisions/](memory/decisions/) = "결정의 reasoning history"

세 종류가 명확히 구분되어야 함.

## 8. 재검토 조건 (옵션 3 자체의 trigger)

- **World Model 분야 새 frontier 등장** (예: World Labs 정식 paper) → Module 3 우선순위 조정
- **회사 업무 우선순위 변화** → 특정 axis 가속 또는 후순위로 이동
- **Track B 진입 후 이론 부족 발견** → 후순위 자료 정독으로 복귀
- **3개월 후** (~Phase 5 종료 시점) → 진도·만족도 점검 + 옵션 재선택

## 9. 빠른 navigation

| 무엇을 알고 싶으면 | 어디로 가나 |
|---|---|
| 현재 어디 / 다음 무엇 | [SESSION_HANDOFF.md](SESSION_HANDOFF.md) |
| 프로젝트 정체성 / 헌법 | [CLAUDE.md](CLAUDE.md) |
| 영구 메모리 전체 카탈로그 | [memory/INDEX.md](memory/INDEX.md) |
| 각 논문 정독 요약 (학습 교재) | [reports/papers/](reports/papers/) |
| 왜 이 결정인가 (reasoning) | [memory/decisions/](memory/decisions/) |
| 정독 중 얻은 비자명한 통찰 | [memory/learnings/](memory/learnings/) |
| 세션별 작업 로그 | [sessions/](sessions/) |
| 사용자 선호·환경 | [memory/preferences/](memory/preferences/) |

## 10. 변경 이력

| Date | Session | 변경 |
|---|---|---|
| 2026-05-18 | session-001 | 초기 Track A 계획 (8편) 수립 |
| 2026-05-18 | session-002 | RT-2 정독 |
| 2026-05-19 | session-003 | OpenVLA 정독 |
| 2026-05-20 | session-004 | SmolVLA 정독 + light-code-checkpoints 정책 |
| 2026-05-20 | session-005 | 갭 분석 → 옵션 3 (확장 커리큘럼) 채택 |
| 2026-05-22 | session-006 | ROADMAP.md 통합 + SmolVLA Action Expert 상세 보강 |
| 2026-05-22 | session-007 | SmolVLA.md 보강 (Tiling/Pixel shuffle/Loss prediction vs target 명확화) |
| 2026-05-23 | session-008 | **Phase 5.5 신설 — LeRobot paper (arXiv:2602.22818) 추가** (Pre-Track-B Infrastructure Reading) |
