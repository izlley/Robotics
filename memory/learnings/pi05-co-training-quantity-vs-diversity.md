---
name: pi05-co-training-quantity-vs-diversity
description: π0.5의 핵심 발견 — Target embodiment(MM) data 2.4%만으로도 cross-embodiment + web + HL + VI co-training 결합 시 unseen home 일반화 가능
metadata:
  type: learning
  created_at: 2026-05-26
  updated_at: 2026-05-26
  source: papers/core-models/π0.5- a Vision-Language-Action Model with Open-World Generalization.pdf
---

# 한 줄 사실

π0.5는 mobile manipulator(MM) 데이터 400h **2.4%만**으로 학습되고, 나머지 **97.6%는 다른 source** (ME / CE / HL / WD)에서 옴. 그럼에도 **학습에 없던 진짜 가정집**에서 10-15분 cleaning task 수행. **Data diversity가 data quantity를 능가**.

# 5+ Co-Training Sources

| 약자 | 의미 | 비중 (Pre-training) |
|---|---|---|
| MM | Mobile Manipulator (target) | **2.4%** |
| ME | Multi-Environment non-mobile robots | 대부분 |
| CE | Cross-Embodiment laboratory robots | 대부분 |
| HL | High-Level subtask prediction | 일정 |
| WD | Web Data (VQA, captioning, localization) | 일정 |
| VI | Verbal Instruction (post-training only) | ~11% of HL subset |

# 정량적 증거 (Ablation)

| 제거된 source | Mock home perf | Language following OOD |
|---|---|---|
| Full π0.5 | 100% baseline | 100% |
| no-WD | ≈ same | **significant drop** ★ |
| no-ME | significantly drop | drop |
| no-CE | significantly drop | drop |
| no-ME&CE | strongly drop | strongly drop |

해석:
- ME + CE는 manipulation generalization에 결정적
- WD는 unseen object 인식에 결정적
- 각 source가 다른 axis에 기여 → 합쳐야 generalization 완성

# Verbal Instruction (VI)의 특별한 가치

11% 비중인데 제거하면 high-level 성능 크게 떨어짐.

**무엇인가**: 전문가가 실시간으로 robot에게 subtask를 말로 지시 → robot이 trained low-level policy로 실행 → 그 sequence가 demo data.

```
Expert: "open the drawer"   -> robot opens
Expert: "pick up mitten"    -> robot picks up
Expert: "put in drawer"     -> robot puts in
```

→ **High-level subtask sequence의 demonstration** 자동 수집.

# 왜 비자명한가

기존 통념:
- "더 많은 robot data = 더 좋은 generalization"
- "Target embodiment data가 대부분이어야 함"

π0.5의 반증:
- **Quantity 부족 (400h, 2.4%)을 diversity로 보충 가능**
- 다양한 robot, 다양한 task, 다양한 modality가 합쳐지면 target embodiment 학습 data 적어도 OK
- 이는 LLM의 Instruction Tuning 발견 (Tülü, FLAN)과 정확히 같은 정신

# 응용 포인트

**우리 스터디 Track c new idea 후보**:

1. **MM data ratio sweep**: 0.5% / 2.4% / 10% / 50% / 90% 학습 → "target data가 어느 정도여야?" sweet spot
2. **Source 추가 효과**: 5개 외에 더 (audio, tactile, language 외 modality)
3. **VI scale up**: 사람 대신 LLM이 자동 verbal instruction 생성
4. **Domain-specific HL training의 가치**: GPT-4 vs in-domain HL 비교 (π0.5 결과 재현)

**LLM 영역과의 1:1 mapping**:
- Tülü, FLAN, T0 (multi-task tuning) → π0.5 co-training
- Constitutional AI / RLHF (human feedback) → VI
- Chain-of-Thought → Hierarchical inference
- Mixture of training tasks → 5+ source mixture

# 한계·미해결

- Source weighting 자동 결정 미구현 (heuristic)
- 새 modality 추가 시 효과 예측 어려움
- VI 같은 high-cost data를 자동화하면 효과가 유지되나?
- Diversity의 한계는 어디인가? 1000 sources?

# 관련 메모리

- [[pi0-pre-post-training]] — π0의 pre/post training 분리 (π0.5의 전신)
- [[rt2-co-fine-tuning-trick]] — 같은 정신, web data 비중 50-66%
- [[pi05-hybrid-fast-flow]]
- [[pi05-hierarchical-inference]]
