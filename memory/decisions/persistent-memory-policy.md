---
name: persistent-memory-policy
description: 본 프로젝트는 project-dir 자체에 memory/sessions를 보존하고 GitHub로 동기화한다
metadata:
  type: decision
  created_at: 2026-05-18
  updated_at: 2026-05-18
---

# 결정

본 프로젝트는 모든 세션 컨텍스트·결정·진행 상태를 **`/workspace/izlley/sllm/Robotics/` 내부**의 `SESSION_HANDOFF.md`, `memory/`, `sessions/`에 보존한다. 추가로 GitHub `izlley/Robotics` repo로 동기화하여 외부 백업을 확보한다.

# 이유

- `/root/.claude/projects/-workspace/memory/`는 pod-local. Pod이 죽거나 새 환경으로 이주하면 손실 위험
- `/workspace/izlley/`는 CephFS 마운트 (`stat -f` 확인됨, type=ceph) → 영구 저장 보장
- 사용자가 명시적으로 "pod이 죽어도 복구 가능하도록 프로젝트 dir에 메모리·이력을 저장해달라"고 요청
- GitHub 동기화는 CephFS 자체 장애·실수 삭제에 대비한 추가 보험

# 구조 (요약 — 자세한 건 [`../README.md`](../README.md))

```
Robotics/
├── SESSION_HANDOFF.md      # 다음 세션 첫 진입점, ≤200줄
├── memory/
│   ├── README.md           # 정책 문서
│   ├── INDEX.md            # 카탈로그
│   ├── decisions/          # 확정 결정
│   ├── learnings/          # 비자명한 통찰
│   ├── preferences/        # 사용자 선호
│   └── references/         # 외부 자료 포인터
└── sessions/
    └── YYYY-MM-DD-session-NNN.md
```

# 갱신 규칙

- 매 세션 종료 시: `sessions/<date>-<seq>.md` 생성 + `SESSION_HANDOFF.md` §3,§4 갱신
- 결정·학습 발생 시 즉시 memory/ 하위에 파일 작성 + `INDEX.md` 한 줄 entry
- 민감 정보 절대 금지 (GitHub public 가능성 고려)

# /root/.claude/...와의 관계

- /root/.claude 메모리: cross-project, 일반적인 사용자 정보, 사내 환경
- 본 프로젝트 memory/: 본 프로젝트 전용 사실
- **중복 허용** — recovery용 이중 백업 목적. 예: user-role.md는 양쪽에 둘 다 존재

# 영향

- 매 turn마다 추가 작업 (~5분): sessions 로그 작성, SESSION_HANDOFF 갱신
- 누적 비용: 메모리 파일 수가 늘면 INDEX 유지가 부담될 수 있음 → 100개 넘어가면 토픽별 sub-index 도입 검토
- GitHub 동기화 책임은 사용자가 직접 commit/push (Claude는 prep만)

# 재검토 조건

- 메모리 파일 수가 너무 많아져 신호 대 잡음 비율이 낮아질 때 → 정리·통합
- /root/.claude 메모리 시스템과의 중복이 혼란을 야기할 때 → 한 쪽으로 통합

# 관련 메모리

- [[pod-environment]] — CephFS 영구 정책의 근거
- [[track-selection]]
- [[baseline-papers-scope]]
