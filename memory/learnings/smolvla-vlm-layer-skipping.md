---
name: smolvla-vlm-layer-skipping
description: VLM의 마지막 N층은 자연어 generation에 특화되어 robot 제어엔 불필요. 앞 L/2층만 써도 성능 유지, 계산 ½
metadata:
  type: learning
  created_at: 2026-05-20
  updated_at: 2026-05-20
  source: papers/core-models/SmolVLA-A-Vision-Language-Action-Model-for-Affordable-and-Efficient-Robotics.pdf
---

# 한 줄 사실

SmolVLA는 SmolLM-2 (SmolVLM-2의 LLM, 16 layer) 중 **앞 8 layer(N=L/2)만 사용**하고 마지막 8개는 버린다. LIBERO 평균 75.0% vs full 78.5% — 3.5pp 손해로 **계산 ½ 절감**. 더 흥미롭게, "큰 VLM의 절반 ≈ 작은 VLM(256M) 통째"라는 trade-off.

# 왜 비자명한가

LLM/VLM 분야의 직관은 "마지막 layer feature가 가장 정제됨". 이 통설이 깨진다:

- El-Nouby (2024), Bolya (2025), Rajasegaran (2025) — "best downstream feature가 마지막 layer가 아닐 수 있다"
- LLM/VLM의 마지막 N층은 **자연어 generation에 특화** (logit projection, next-token prediction)
- Robot 제어에 필요한 건 **semantic + spatial perception** → 중간층(layer 8 정도)에서 이미 충분

이 통찰은 VLA만이 아니라 **모든 perception-only downstream task**에 적용 가능.

# 정량 데이터 (Table 8, LIBERO Avg)

| Layer 사용 | LIBERO Avg | 의미 |
|---|---|---|
| 8 (=L/2) | 75.0 | 큰 VLM의 절반 |
| 16 (=L) | 78.5 | 큰 VLM 전체 |
| Skip %2 (격수 8층) | 75.5 | 균등 sampling |
| **VLM-256M (full)** | **75.8** | 작은 VLM 통째 |

→ **"큰 모델 절반" ≈ "작은 모델 통째"** — pruning > size-down 결론.

# 메커니즘 (왜 작동하나)

VLM의 layer별 기능 (LLM 연구에서 알려진 것):
- **초기 layer (1-N/4)**: low-level visual/syntactic features
- **중간 layer (N/4 - 3N/4)**: high-level abstractions (semantic, spatial) ← **perception에 핵심**
- **마지막 layer (3N/4 - L)**: task-specific (generation, logit projection) ← **action에는 redundant**

따라서 VLA처럼 LLM의 generation을 안 쓰고 features만 쓰는 모델은 마지막 layer 무용지물.

# 응용 포인트

**우리 스터디 활용**:
- 다른 VLM(예: Qwen-VL, InternVL)에 같은 trick 적용 가능
- N sweep ablation은 trivial한 new idea 후보 (Track c)
- 더 작은 모델 만들 때 일반 원칙으로 활용

**확장 idea**:
- "early exit" 메커니즘 — runtime에 input에 따라 N을 조정
- VLM 위에 작은 robotics-specific head를 추가하면 (linear probe) 더 효율 가능

# 한계·미해결

- N의 sweet spot이 모델·task별로 다를 수 있음 (SmolVLM-2에서는 N=L/2가 적절, 다른 VLM은?)
- "마지막 layer 무용" 가설은 작은 LLM에서 검증됨. 큰 LLM(70B+)에서도 같은가?
- Robot task가 추상화될수록 (long-horizon planning) 마지막 layer 정보가 필요할 수도

# LLM 도구와의 관계

| LLM 영역 | 유사 도구 |
|---|---|
| Model distillation | "큰 모델 절반"이 distillation 결과와 유사 |
| Early exit (DeeBERT, FastBERT) | 같은 아이디어를 runtime에 동적으로 |
| Layer pruning | Static pruning of last layers |
| Linear probe | 중간 layer features를 task-specific head로 |

→ SmolVLA는 layer pruning을 robotics에 적용한 첫 명시적 사례.

# 관련 메모리

- [[openvla-dual-vision-encoder-fusion]] — OpenVLA는 vision encoder 강화 방향
- [[summary-style]]
