# SESSION_HANDOFF — 현재 상태 (Current State)

> **다음 세션 첫 명령**: 이 파일을 먼저 읽고, 학습 계획은 [`ROADMAP.md`](ROADMAP.md) 참조. 이어서 [`memory/INDEX.md`](memory/INDEX.md) → 가장 최근 `sessions/*.md` 1~2개 순서로 읽어 컨텍스트 복원하세요.
>
> **Last updated**: 2026-05-22 (session-006 종료 시점)
>
> **이 파일의 역할**: "지금 어디" 만 짧게. 자세한 계획은 [ROADMAP.md](ROADMAP.md), 결정 이력은 [memory/decisions/](memory/decisions/).

---

## 1. 프로젝트 한 줄 정의

`/workspace/izlley/sllm/Robotics` — **VLA/World Model/RFM 학습 self-curriculum**, dnotitia LLM 엔지니어 정용엽이 회사 RFM R&D 책무를 위해 진행. GitHub `izlley/Robotics` repo 동기화.

## 2. 현재 위치 (Quick Status)

- **Phase**: Track A — Phase 3 (핵심 8편 정독)
- **진도**: **3/8 완료** (RT-2, OpenVLA, SmolVLA)
- **다음 turn**: **π0 정독** (보유 PDF, 다운로드 불필요)
- **로드맵**: [`ROADMAP.md`](ROADMAP.md) (옵션 3, 균형 확장 확정)

## 3. 최근 완료 (시간 역순)

- session-006 (2026-05-22): **`ROADMAP.md` 신설** (학습 계획 통합) + SmolVLA.md Action Expert 섹션 상세 보강 (per-block 구조, VLM feature flow, tensor shapes, AdaLN τ conditioning)
- session-005 (2026-05-20): 갭 분석 → **옵션 3 (확장 커리큘럼) 채택** ([extended-curriculum-for-rfm-research.md](memory/decisions/extended-curriculum-for-rfm-research.md))
- session-004 (2026-05-20): SmolVLA 정독 + light-code-checkpoints 정책
- session-003 (2026-05-19): OpenVLA 정독 + 3 learnings
- session-002 (2026-05-18): RT-2 정독 + 2 learnings
- session-001 (2026-05-18): 프로젝트 인프라 + git init + initial commit/push

## 4. 다음 즉시 작업

1. **이번 session-006 commit + push** — ROADMAP.md 신설, SmolVLA.md 보강, SESSION_HANDOFF·CLAUDE·README 갱신, session-006 로그
2. **사용자 학습 시간** — SmolVLA.md §3.4.3 (Action Expert 상세) 읽고 flow matching paradigm internalize. ROADMAP.md로 전체 그림 한 번 확인.
3. **Phase 3 다음 정독: π0** (Black 2024, PI) — SmolVLA의 큰 형. Flow matching action expert paradigm의 robotics 원조. 보유 PDF.
4. 그 후 Phase 3 잔여: π0.5 → π★0.6 → π0.7 → GR00T N1
5. **Light 체크인 후보** (사용자 요청 시): OpenVLA / SmolVLA inference smoke test

자세한 후속 phase 흐름: [`ROADMAP.md` §3](ROADMAP.md).

## 5. 미해결 / 검토 대기

- **⚠️ 보안: PAT rotate 필요** — 과거 PAT 노출. GitHub Settings → Developer settings → PAT → Revoke + 재발급.
- 향후 push 인증: origin URL에 PAT 없음. 매 push마다 `.keys` 읽기.

## 6. 핵심 파일 cheat sheet

```
/workspace/izlley/sllm/Robotics/
├── ROADMAP.md                   # 학습 계획 single source of truth ★
├── SESSION_HANDOFF.md           # 본 파일 (current state)
├── CLAUDE.md                    # 프로젝트 헌법
├── README.md                    # GitHub 노출용
├── papers/core-models/          # PDF — RT-2, OpenVLA, SmolVLA, π0 시리즈
├── reports/
│   ├── VLA-study-init.md        # 25KB seed
│   ├── papers/                  # 정독 요약 (학습 교재): RT-2.md, OpenVLA.md, SmolVLA.md
│   └── synthesis/               # 이론 종합 (Phase 5에서 채워질 예정)
├── memory/
│   ├── INDEX.md                 # 카탈로그
│   ├── decisions/               # 7건 (track, scope, persistent-memory, extended-curriculum 등)
│   ├── learnings/               # 8건 (RT-2, OpenVLA, SmolVLA 각각의 비자명한 통찰)
│   ├── preferences/             # 4건 (user-role, pod-env, summary-style, light-code-checkpoints)
│   └── references/              # (비어 있음, 외부 자료 포인터용)
└── sessions/                    # 1~6번 로그
```

H200 클러스터 접근:
- `kubectl --kubeconfig=/workspace/izlley/sllm/open-r1-6/env/ncloud-vlm-kubeconfig.yaml exec qwen3-122b-0 -c qwen3-122b -- <cmd>`

## 7. 재시작 시 가장 먼저 읽을 메모리

1. **[`ROADMAP.md`](ROADMAP.md)** — 전체 계획 한눈에
2. [`memory/decisions/extended-curriculum-for-rfm-research.md`](memory/decisions/extended-curriculum-for-rfm-research.md) — 왜 이 계획인지
3. [`memory/preferences/summary-style.md`](memory/preferences/summary-style.md) — 요약 스타일 (md=학습 교재)
4. [`memory/preferences/light-code-checkpoints.md`](memory/preferences/light-code-checkpoints.md) — Light 체크인 정책
5. 직전 sessions/*.md 1~2개

## 8. 환경 핵심 제약 (요약)

- `/workspace/izlley/` = CephFS 영구. `/workspace/data` = pod-local 비영구.
- 122B+ 학습 시 offload 금지 (과거 OOM)
- H200은 ssh 불가, `kubectl exec` 경유
- LR 선택은 historical track 우선
- GitHub `izlley/Robotics` repo 동기화 — 민감 정보 금지

자세히: [`memory/preferences/pod-environment.md`](memory/preferences/pod-environment.md)
