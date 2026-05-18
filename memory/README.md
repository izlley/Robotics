# memory/ — 프로젝트 영구 메모리

이 디렉토리는 Robotics 스터디 프로젝트의 **세션 간 영구 메모리**다. Pod 재시작·새 환경 이주·새 Claude 세션에서도 컨텍스트를 복원할 수 있도록 프로젝트 디렉토리 자체에 저장된다.

`/root/.claude/projects/-workspace/memory/`(pod-local, 비영구 위험)와 별개이며 이중 백업 역할을 한다. 본 디렉토리가 단일 source of truth.

## 구조

```
memory/
├── README.md           # 본 파일 — 정책 가이드
├── INDEX.md            # 모든 메모리 파일 카탈로그 (한 줄 entry)
├── decisions/          # 확정된 결정 사항 (트랙·모델·스코프 등, 잘 안 바뀜)
├── learnings/          # 학습 중 발견한 비자명한 사실·통찰
├── preferences/        # 사용자 선호 (요약 스타일·깊이·문서 길이 등)
└── references/         # 외부 시스템 포인터 (URL·Slack·대시보드 등)
```

## 메모리 파일 형식

각 메모리 파일은 frontmatter + 본문 구조:

```markdown
---
name: <short-kebab-case-slug>
description: <one-line summary — 미래의 자신/Claude가 관련성을 빠르게 판단>
metadata:
  type: <decision | learning | preference | reference>
  created_at: <YYYY-MM-DD>
  updated_at: <YYYY-MM-DD>
---

<본문 — type별 구조 권장>
```

### type별 본문 구조 권장

- **decision**: 결정 / 이유 / 대안 / 영향 / 재검토 조건
- **learning**: 사실 한 줄 / 출처(논문·실험) / 왜 비자명한가 / 응용 포인트
- **preference**: 선호 사항 / 왜 (계기·과거 incident) / 적용 시점
- **reference**: 무엇 / 위치(URL·경로) / 언제 사용

## 갱신 규칙

1. **세션 중 결정 발생 시**: 즉시 `decisions/<slug>.md` 작성. 큰 결정은 `../SESSION_HANDOFF.md`에도 한 줄 반영.
2. **세션 중 비자명한 학습 발생 시**: 즉시 `learnings/<slug>.md` 작성.
3. **세션 종료 시**: 신규 추가된 파일을 `INDEX.md`에 한 줄로 카탈로그.
4. **메모리가 outdated**: 발견 즉시 갱신 또는 삭제. 잘못된 메모리는 없는 것보다 나쁘다.

## 무엇을 저장하지 않는가

- API 키·비밀번호·민감 정보 (GitHub 동기화 대상이므로 절대 금지)
- 코드에서 바로 확인 가능한 사실 (디렉토리 구조, 함수 시그니처, 의존성 버전 등) — 메모리가 아닌 코드/문서를 읽으면 된다
- 일시적 작업 상태 — 그건 `../sessions/`에 기록
- 논문 요약 자체 — 그건 `../reports/papers/`에 기록 (메모리는 *논문에서 깨달은 비자명한 통찰만*)

## 링크 표기

다른 메모리 파일 참조 시 `[[name]]` 또는 마크다운 링크 사용:
- `[[track-selection]]` — slug 기반
- `[track-selection](decisions/track-selection.md)` — 클릭 가능

## 복구 시나리오 (Pod 사망 → 새 환경)

```bash
# 1. SESSION_HANDOFF로 현재 상태 파악
cat /workspace/izlley/sllm/Robotics/SESSION_HANDOFF.md

# 2. 메모리 카탈로그 훑기
cat /workspace/izlley/sllm/Robotics/memory/INDEX.md

# 3. 우선순위 파일 읽기 (SESSION_HANDOFF §7 참조)
cat /workspace/izlley/sllm/Robotics/memory/decisions/*.md

# 4. 직전 세션 로그 읽기
ls -t /workspace/izlley/sllm/Robotics/sessions/ | head -2 | xargs -I {} cat /workspace/izlley/sllm/Robotics/sessions/{}
```

→ 새 세션 즉시 작업 재개 가능.
