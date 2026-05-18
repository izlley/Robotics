# Robotics — VLA 모델 중심 로보틱스 심화 스터디

> Personal study repository for Robotics — focused on **Vision-Language-Action (VLA) foundation models**.
> Owner: 정용엽 (dnotitia)

## 목표

Robotics 분야의 frontier 기술, 특히 VLA(Vision-Language-Action) 모델에 대해 다음 4단계로 심도있게 학습한다.

1. **기본 개념과 이론 학습** — 논문 정독 + 트렌드 정리
2. **오픈소스 활용 구현·학습·평가 실습** — LeRobot / OpenVLA / openpi 등
3. **new idea 실험** — data·architecture·training·evaluation·inference 측면
4. frontier 도달까지 반복

현재 페이즈: **(1) 이론 학습** — Track A 진행 중.

## 디렉토리 안내

| 경로 | 역할 |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | Claude Code/AI 어시스턴트용 프로젝트 헌법 |
| [`SESSION_HANDOFF.md`](SESSION_HANDOFF.md) | 작업 재개 시 첫 진입점 (현재 트랙·다음 작업) |
| [`papers/`](papers/) | 논문 PDF (카테고리별 분류) |
| [`reports/`](reports/) | 학습 산출물 — seed 문서, 논문 요약, 통합 분석 |
| [`memory/`](memory/) | 영구 메모리 — 결정·통찰·선호·외부 참조 |
| [`sessions/`](sessions/) | 세션별 작업 로그 |

## 추천 진입 경로

처음 이 repo를 보는 경우:

1. **트렌드 한 눈에 보기**: [`reports/VLA-study-init.md`](reports/VLA-study-init.md) — VLA 모델 13종 비교 + 7대 트렌드 (25KB)
2. **현재 상태 확인**: [`SESSION_HANDOFF.md`](SESSION_HANDOFF.md)
3. **개별 모델 깊이 학습**: [`reports/papers/`](reports/papers/) — 9-섹션 표준 요약
4. **결정 히스토리**: [`memory/decisions/`](memory/decisions/)

## 학습 대상 모델 (Track A 스코프)

### 핵심 8편 — 풀 9-섹션 요약

1. **RT-2** (Google DeepMind, 2023) — VLA 패러다임 정립
2. **OpenVLA** (Stanford/UCB, 2024) — 7B 오픈소스 표준
3. **SmolVLA** (HF/LeRobot, 2025) — 450M 경량
4. **π0** (Physical Intelligence, 2024) — flow matching action expert
5. **π0.5** (PI, 2025) — open-world generalization
6. **π★0.6** (PI, 2026) — RL self-improvement
7. **π0.7** (PI, 2026) — steerable / compositional
8. **GR00T N1** (NVIDIA, 2025) — humanoid foundation

### 2nd tier / 기초 기술 / 데이터 / Survey

자세한 스코프: [`memory/decisions/baseline-papers-scope.md`](memory/decisions/baseline-papers-scope.md)

## 영구 메모리 정책

본 repo는 pod 재시작·환경 이주에도 작업을 재개할 수 있도록 **모든 컨텍스트를 repo 자체에 보존**한다.

- 매 세션 종료 시 [`sessions/`](sessions/)에 로그 추가
- 결정·통찰 발생 시 [`memory/`](memory/) 하위에 frontmatter 형식 파일로 즉시 기록
- [`SESSION_HANDOFF.md`](SESSION_HANDOFF.md)는 항상 최신 상태 유지

자세한 정책: [`memory/README.md`](memory/README.md)

## License

Personal study materials. PDF 논문 자체는 각 저작권자(arXiv 저자, NVIDIA, Hugging Face, Physical Intelligence 등)에 귀속된다. 본 repo의 요약·정리 문서는 학습 목적의 paraphrase·summarization이다.

## Contact

jungyup2@dnotitia.com
