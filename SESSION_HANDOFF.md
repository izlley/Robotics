# SESSION_HANDOFF — Robotics 스터디 프로젝트

> **다음 세션 첫 명령**: 이 파일을 먼저 읽고, 이어서 [`memory/INDEX.md`](memory/INDEX.md) → 가장 최근 `sessions/*.md` 1~2개 순서로 읽어 컨텍스트를 복원하세요.
>
> **Last updated**: 2026-05-18 (session-001 종료 시점)

---

## 1. 프로젝트 한 줄 정의

`/workspace/izlley/sllm/Robotics` — **VLA(Vision-Language-Action) 모델 중심의 로보틱스 심화 스터디 프로젝트**. dnotitia LLM 엔지니어 정용엽이 LLM/VLM 경험을 기반으로 로보틱스 frontier 기술을 습득하기 위한 트랙. GitHub `izlley/Robotics` repo로 동기화.

## 2. 현재 트랙 & 페이즈

- **상위 트랙**: 학습 절차 (a) "기본 개념·이론 학습"
- **세부 트랙**: Track A — 이론 심화 (논문 정독)
- **현재 페이즈**: Phase 0·1 + Git prep 완료. **다음은 Phase 2 (자료 수집)**
- **전체 플랜 파일**: `/root/.claude/plans/workspace-izlley-sllm-robotics-rustling-snowflake.md` (pod-local, 비영구 위험. 본 SESSION_HANDOFF가 복구의 source of truth)

## 3. 최근 완료된 작업

- [x] 프로젝트 디렉토리 구조·seed 문서·보유 논문 파악
- [x] Track A(이론 심화) + baseline 4편(RT-2, OpenVLA, SmolVLA, GR00T N1) + 확장 자료 수집 결정
- [x] CephFS 영구성 확인 (`/workspace/izlley` = ceph 마운트)
- [x] Phase 0: 영구 메모리 인프라 구축 (`memory/`, `sessions/`, SESSION_HANDOFF, INDEX, 5개 메모리 파일, CLAUDE.md 개정)
- [x] Phase 1: `papers/{core-models,related-models,foundations,data-benchmarks,surveys}/` 생성 + 기존 π0 4편 → `papers/core-models/` 이동
- [x] Phase 1: `reports/{papers,external,synthesis}/` 생성 + `reports/papers/_template.md` (9-섹션) 작성
- [x] Git prep: `git init -b main` + `.gitignore` + `README.md` + remote `origin = https://github.com/izlley/Robotics.git` 설정

## 4. 다음 즉시 작업 (이어서 진행)

1. **사용자가 직접 수행할 액션**: 
   - `cd /workspace/izlley/sllm/Robotics && git add -A && git commit -m "Initial commit ..." && git push -u origin main`
   - (또는 Claude에게 명시적으로 commit/push 요청)
2. **Phase 2**: 1차 batch 자료 다운로드 — RT-2 / OpenVLA / SmolVLA / GR00T N1 (PDF는 `papers/core-models/`) + Diffusion Policy / ACT (→ `papers/foundations/`) + Open X-Embodiment (→ `papers/data-benchmarks/`) + VLA survey 1편 (→ `papers/surveys/`)
3. **Phase 3 시작**: RT-2 정독 → `reports/papers/RT-2.md` 작성 (워크플로우 검증)

## 5. 미해결 질문 / 결정 대기 항목

- 없음 (현재 트랙 시작 직후)

## 6. 핵심 파일·경로 cheat sheet

```
Project root: /workspace/izlley/sllm/Robotics
├── CLAUDE.md                                    # 프로젝트 헌법
├── SESSION_HANDOFF.md                           # 이 파일
├── papers/                                      # 논문 PDF (보유 4편 — π0 시리즈)
├── reports/VLA-study-init.md                    # seed 문서 (25KB VLA 트렌드)
├── memory/
│   ├── INDEX.md                                 # 메모리 카탈로그
│   ├── decisions/                               # 확정 결정
│   ├── learnings/                               # 비자명한 사실
│   ├── preferences/                             # 사용자 선호
│   └── references/                              # 외부 자료 포인터
└── sessions/                                    # 세션 로그

Useful commands (H200 클러스터 작업 시):
- kubectl --kubeconfig=open-r1-6/env/ncloud-vlm-kubeconfig.yaml exec qwen3-122b-0 -c qwen3-122b -- ...
```

## 7. 재시작 시 가장 먼저 읽을 메모리 파일

1. [`memory/decisions/track-selection.md`](memory/decisions/track-selection.md) — 왜 Track A를 선택했는가
2. [`memory/decisions/baseline-papers-scope.md`](memory/decisions/baseline-papers-scope.md) — 어떤 논문을 어떤 순서로 보는가
3. [`memory/decisions/persistent-memory-policy.md`](memory/decisions/persistent-memory-policy.md) — 영구 메모리 정책 자체
4. [`memory/preferences/user-role.md`](memory/preferences/user-role.md) — 사용자 배경
5. [`memory/preferences/pod-environment.md`](memory/preferences/pod-environment.md) — H200 클러스터·CephFS 제약

## 8. 환경 핵심 제약 (요약)

- `/workspace/izlley/` = CephFS 영구. `/workspace/data` = pod-local 비영구.
- 122B+ 학습 시 offload 금지 (과거 OOM)
- H200은 ssh 불가, `kubectl exec` 경유
- LR 선택은 historical track 우선, unproven 점프 금지
- 본 프로젝트 dir은 GitHub `izlley/Robotics` repo에 동기화될 예정 — 민감 정보 금지

## 9. 갱신 규칙

- 매 작업 turn 종료 시 본 파일의 §3, §4, §7 갱신
- 큰 결정·학습은 `memory/` 하위에 별도 파일로 즉시 기록 후 INDEX 업데이트
- 본 파일은 **200줄 이내 유지** — 자세한 내용은 memory/ 또는 sessions/로 분기
