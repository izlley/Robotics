---
name: light-code-checkpoints
description: 이론 정독 사이에 5~10분 분량 "light 코드 체크인"을 끼워 넣어 감각·작동 확인 → 본격 hands-on(Track B)은 Track A 종료 후
metadata:
  type: preference
  created_at: 2026-05-20
  updated_at: 2026-05-20
---

# 정책

이론 정독 트랙(Option C, 8편 + synthesis) 도중에 **5~10분짜리 가벼운 코드 체크인**을 끼워 넣는다. 본격 환경 setup·LoRA fine-tune 같은 무거운 작업은 Track A 완료 후로 미루되, "이 모델이 실제로 돌아가는구나"라는 감각만 유지하는 게 목적.

# Why (계기)

2026-05-20 session-004(SmolVLA) 종료 직전에 사용자가 제안:
> "이론 정독 중에도 5분 분량의 light '코드 헤어드라이어' 이 아이디어는 매우 좋은것 같습니다."

(이전 turn에서 Option C 추천 시 부차 옵션으로 제시한 아이디어를 사용자가 채택)

# How to apply

## 기본 원칙

- **5~10분 이내**: 환경 setup 없음, 코드 한 두 cell 만
- **목표는 "감각" 유지**: model.from_pretrained + 1회 inference + 결과 확인. 학습 X, 평가 X.
- **이론 흐름 단절 금지**: 같은 turn에 정독·요약·체크인 모두 하지 말고, 별도 짧은 turn으로 분리하거나 정독 turn 마지막에 5분만 끼우기
- **결과는 sessions/ 로그에 기록**: "OpenVLA-7b가 4090 16GB에서 inference 1회 6 Hz로 작동 확인" 같은 한 줄

## 체크인 후보 (정독 진도별)

| 정독 완료 시점 | Light check 후보 | 예상 명령 |
|---|---|---|
| OpenVLA (✓ 완료) | OpenVLA HF weights load + dummy image inference | `transformers.AutoModel.from_pretrained("openvla/openvla-7b")` + 1회 forward |
| SmolVLA (✓ 완료) | SmolVLA HF weights load + dummy image inference | `LeRobotModel.from_pretrained("lerobot/smolvla_base")` + 1회 forward |
| π0 정독 후 | π0 (Black 2024) weight 또는 openpi inference | openpi codebase clone + sample run |
| π★0.6 정독 후 | RECAP 코드 구조만 확인 (가능하다면) | repo structure browse |
| GR00T N1 정독 후 | NVIDIA Isaac GR00T weights 확인 | HF 또는 NGC에서 weight pull |

## 무엇을 체크인이라 부르지 않는가 (= Track B로 미룸)

- LeRobot 전체 환경 setup
- Simulator (LIBERO, RoboCasa) 설치 + 사용
- Fine-tuning (LoRA든 full이든)
- Multi-task evaluation
- 실 robot 연결

이들은 환경 의존성·시간이 5분 한참 넘어가므로 Track A 종료 후 hands-on session에서.

## 환경 가정

- 본 프로젝트는 dnotitia H200 클러스터 접근. 가벼운 inference는 H200 1 GPU로 충분.
- 모델 weight download는 HuggingFace (인터넷 OK)
- 영구 저장은 CephFS (`/workspace/izlley/sllm/Robotics/code/` 또는 별도 path)
- 추가 가능: 본 프로젝트 dir 안에 `code/light_checks/` 디렉토리 두고 `.py` 또는 `.ipynb` 파일 누적

## 산출물

각 체크인은 다음을 남긴다:
1. **Code**: `Robotics/code/light_checks/<model>_smoke_test.py` (또는 .ipynb)
2. **Session log**: `sessions/<date>-session-NNN.md`에 5줄 정도 기록 ("model X loaded, 1 inference OK, latency Y ms, VRAM Z GB")
3. (선택) **Learnings**: 비자명한 발견이 있으면 `memory/learnings/`에 기록

# 관련 메모리

- [[summary-style]] — 이론 학습 스타일 (md가 메인 교재). Light check는 이걸 보완
- [[user-role]] — LLM 엔지니어이므로 코드 자체는 친숙. 부담 적음
- [[pod-environment]] — H200 접근 방법
- [[track-selection]] — Option C(이론 먼저) 결정. Light check는 그 원칙을 깨지 않는 보조 활동
