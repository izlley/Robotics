---
name: track-selection
description: 학습 절차 (a)의 첫 세부 트랙으로 Track A(이론 심화 — 논문 정독)를 선택
metadata:
  type: decision
  created_at: 2026-05-18
  updated_at: 2026-05-18
---

# 결정

학습 절차 (a) "기본 개념·이론 학습"의 진입 트랙으로 **Track A: 이론 심화 (논문 정독)**를 선택한다.

대안이었던 Track B(오픈소스 환경 셋업)와 Track C(seed 문서 보강 + 토픽 분기)는 Track A 이후로 보류.

# 이유

- 보유 자료(π0 시리즈 4편)가 이미 이론 심화에 최적화된 형태로 준비되어 있음
- VLA·로보틱스 도메인이 사용자에게 새로운 분야이므로 코드부터 만지기 전에 **개념·아키텍처·평가 패러다임을 충분히 internalize**할 필요가 있음
- seed 문서([../../reports/VLA-study-init.md](../../reports/VLA-study-init.md))가 이미 트렌드 큰 그림을 잡아주고 있으므로, 다음 단계는 모델별 깊이 파기
- Track B 환경 셋업은 LeRobot/openpi/OpenVLA 중 어느 stack을 쓸지부터 결정해야 하는데, 그 결정 자체가 이론 심화 결과에 의존적

# 대안

| 트랙 | 왜 지금은 아님 |
|---|---|
| Track B (환경 셋업) | 모델·스택 선택이 이론 정독 결과에 의존. 먼저 정독해야 합리적 선택 가능 |
| Track C (seed 보강) | seed 문서가 이미 25KB로 충분. 보강은 정독 후 synthesis 단계에서 더 효율적 |

# 영향

- Phase 2(자료 수집) → Phase 3(8편 정독 + 보조 자료) → Phase 4(synthesis 3편) 순차 진행
- 약 4~8주 예상 (논문당 1 turn × 8편 + interleave 자료 + synthesis)
- 이후 자연스럽게 학습 절차 (b) "오픈소스 실습"으로 연결

# 재검토 조건

- 정독 도중 "더 이상 이론으로 얻을 인사이트가 없다"는 판단이 들면 Track B로 조기 전환
- 사용자가 즉각적 실습 필요성을 명시적으로 표명할 경우
- 새로운 frontier 논문이 갑자기 등장해 우선순위 재조정이 필요한 경우

# 관련 메모리

- [[baseline-papers-scope]] — 어떤 논문을 어떤 순서로 볼지
- [[persistent-memory-policy]] — 본 결정을 어디에 어떻게 기록할지
