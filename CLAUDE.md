# Project: Study Robotics Fundamentals

## 0. 다음 세션 진입 가이드 (가장 먼저 읽기)

새 세션 시작 시 다음 순서로 컨텍스트를 복원하세요:

1. [`SESSION_HANDOFF.md`](SESSION_HANDOFF.md) — 현재 작업 상태 (≤200 줄, 항상 최신)
2. [`ROADMAP.md`](ROADMAP.md) — **학습 계획 single source of truth** ★
3. [`memory/INDEX.md`](memory/INDEX.md) — 모든 영구 메모리 카탈로그
4. `memory/decisions/`, `memory/preferences/` 우선순위 파일 (SESSION_HANDOFF §7 참조)
5. 가장 최근 `sessions/*.md` 1~2개 — 직전 세션 로그

## 1. 프로젝트 목표

Robotics 관련 기술들에 대한 심도있는 스터디를 진행한다. 특히 **VLA(Vision-Language-Action) 모델**에 대한 학습을 메인으로 진행한다.

## 2. 학습 절차

a. 기본적인 개념과 이론 학습
b. opensource를 활용한 구현/학습/평가 실습
c. data, 모델 구조적, 학습 방법론, 평가 방법론, 추론 방법론 측면에서 다양한 new idea에 대한 실험 진행
d. 위 학습 과정을 frontier 기술을 습득할때까지 반복한다.

## 3. 디렉토리 구조

```
Robotics/
├── CLAUDE.md                # 본 파일 (프로젝트 헌법)
├── ROADMAP.md               # 학습 계획 single source of truth ★
├── SESSION_HANDOFF.md       # 현재 작업 상태 (turn-by-turn)
├── README.md                # GitHub 노출용
├── papers/                  # 논문 PDF (카테고리별 sub-dir)
│   ├── core-models/         # 핵심 8편
│   ├── related-models/      # 2nd tier
│   ├── foundations/         # 기초 기술 (Diffusion Policy, ACT, FAST 등)
│   ├── data-benchmarks/     # Open X-Embodiment, DROID, LIBERO 등
│   └── surveys/             # Survey 논문
├── reports/                 # 학습 산출물
│   ├── VLA-study-init.md    # 25KB seed 문서 (트렌드 분석)
│   ├── papers/              # 논문별 요약 (9-섹션 템플릿)
│   ├── external/            # 외부 blog·tech report 캡처
│   └── synthesis/           # 통합 분석
├── memory/                  # 프로젝트 영구 메모리 (recovery-safe)
│   ├── README.md            # 정책
│   ├── INDEX.md             # 카탈로그
│   ├── decisions/
│   ├── learnings/
│   ├── preferences/
│   └── references/
└── sessions/                # 세션 로그
```

## 4. 문서화 규칙

- **논문**: `./papers/<category>/` 에 다운로드. 카테고리는 [`memory/decisions/baseline-papers-scope.md`](memory/decisions/baseline-papers-scope.md) 참조.
- **요약**: 각 논문에 대해 `./reports/papers/<model>.md` 작성. 핵심 모델은 9-섹션 풀 템플릿 (`reports/papers/_template.md`), 기초·survey는 간략 요약 가능.
- **외부 자료**: blog post 등은 `./reports/external/<source>-<title>.md`에 markdown 캡처 (URL·fetch 일시 명기).
- **Seed 문서**: [`reports/VLA-study-init.md`](reports/VLA-study-init.md) — 학습 정보 seed. 정독 후 `reports/synthesis/`로 보강.

## 5. 영구 메모리 정책 (Pod 재시작 복구 대비)

본 프로젝트는 모든 컨텍스트·결정·진행 상태를 프로젝트 디렉토리 내부에 저장하여, pod 사망·환경 이주에도 복구 가능하도록 한다. CephFS 마운트(`/workspace/izlley`)에 위치하므로 영구 보장.

- **SESSION_HANDOFF.md**: 세션 시작 시 첫 진입점, 항상 최신 상태로 유지
- **memory/**: 결정·통찰·선호·외부 참조를 frontmatter 형식 파일로 분류 보관
- **sessions/**: 매 작업 세션 종료 시 로그 작성 (`YYYY-MM-DD-session-NNN.md`)

자세한 정책: [`memory/README.md`](memory/README.md), [`memory/decisions/persistent-memory-policy.md`](memory/decisions/persistent-memory-policy.md)

### 갱신 규칙 (Claude에게)

- 매 작업 turn 종료 시: `SESSION_HANDOFF.md` §3, §4, §7 갱신 + `sessions/` 로그 추가
- 결정/학습 발생 시 즉시 `memory/<type>/<slug>.md` 작성 + `memory/INDEX.md`에 한 줄 entry
- 본 파일 §5 정책에 변경 발생 시 `memory/decisions/persistent-memory-policy.md` 함께 갱신

## 6. GitHub 동기화

본 디렉토리는 GitHub `izlley/Robotics` repo에 push되어 외부 백업으로 사용된다.

- **민감 정보 금지**: API 키·비밀번호·사내 dashboard URL 등 절대 commit 하지 않는다
- **대용량 파일**: PDF는 현재 직접 commit. 총량 1GB 넘어가면 Git LFS 도입 검토
- **commit 권한**: 사용자가 직접 수행 (Claude는 prep만; 명시 요청 없으면 commit/push 금지)

## 7. 환경 제약 요약

- `/workspace/izlley` = CephFS 영구. `/workspace/data` = pod-local 비영구.
- H200 클러스터: ssh 불가, `kubectl exec` 경유
- 대형 모델 학습 시 offload 금지 (과거 OOM)
- 자세한 사항: [`memory/preferences/pod-environment.md`](memory/preferences/pod-environment.md)
