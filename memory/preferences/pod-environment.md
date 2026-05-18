---
name: pod-environment
description: Pod·H200 클러스터·CephFS 등 인프라 환경 제약 사항
metadata:
  type: preference
  created_at: 2026-05-18
  updated_at: 2026-05-18
---

# Storage

| 경로 | 종류 | 영구성 |
|---|---|---|
| `/workspace/izlley/` | CephFS (ceph mount, mds_namespace=cephfs) | **영구** ✅ |
| `/workspace/slm/`, `/workspace/aidata/`, `/workspace/backup/` | CephFS | **영구** ✅ |
| `/workspace/data/` | pod-local overlay | **비영구** ❌ |
| `/root/`, `/tmp/`, `/home/` (대부분) | pod-local | **비영구** ❌ |

**Why**: Pod 재시작 시 overlay filesystem이 소실됨. 학습 데이터·체크포인트·메모리·이력은 반드시 CephFS 경로에 두어야 한다.

**How to apply**: 본 프로젝트는 `/workspace/izlley/sllm/Robotics/` 하위에 모든 영구 자료를 보관 (논문, reports, memory, sessions, code).

# Compute (H200 cluster)

- 접근: SSH 불가. `kubectl exec` 경유.
- 명령 예: `kubectl --kubeconfig=open-r1-6/env/ncloud-vlm-kubeconfig.yaml exec qwen3-122b-0 -c qwen3-122b -- <cmd>`
- VLA 모델 (450M~7B)은 H200 1~2 GPU로 충분, 122B 클래스 같은 자원은 불필요할 가능성 높음

# 학습 운영 규칙

- **122B+ offload 금지** — fsdp_offload_params, zero3 cpu offload 시 과거 OOM 경험
- **LR 선택**: historical track 우선, unproven 값 점프 금지 (baseline gate-pass lr 근방 우선)
- **체크포인트·로그**: CephFS에 저장. 자동 cleanup script로 ckpt-XX만 남기는 패턴 사용 가능

# Network

- 외부 인터넷 접근: WebFetch 가능 (HuggingFace, arXiv, GitHub 공개 자원 다운로드 가능)
- 사내 망 리소스: 필요 시 별도 확인

# Persistent Memory (본 프로젝트 특화)

본 프로젝트는 위 제약을 반영해 `Robotics/SESSION_HANDOFF.md`, `Robotics/memory/`, `Robotics/sessions/`에 영구 메모리를 둔다. 자세한 정책은 [[persistent-memory-policy]].

# 관련 메모리

- [[user-role]]
- [[persistent-memory-policy]]
