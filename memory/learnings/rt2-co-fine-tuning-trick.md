---
name: rt2-co-fine-tuning-trick
description: VLM을 robot data로 fine-tune할 때 web data를 50~66% 비중으로 함께 흘려야 web 지식이 보존된다
metadata:
  type: learning
  created_at: 2026-05-18
  updated_at: 2026-05-18
  source: papers/core-models/RT-2-Vision-Language-Action-Models-Transfer-Web-Knowledge-to-Robotic-Control.pdf
---

# 한 줄 사실

VLM을 robot data만으로 fine-tuning(=일반적 fine-tuning)하는 것보다, **web 데이터를 함께 mix해서 학습**하면 unseen object/symbol/reasoning에서 **+10pp 이상** generalization 향상. RT-2 논문이 ablation으로 정량 입증.

# 데이터

| 학습 방식 | 5B (avg unseen) | 55B (avg unseen) |
|---|---|---|
| from scratch | 9% | (skip) |
| fine-tuning only | 42% | 52% |
| **co-fine-tuning** | 44% | **63%** |

Web:Robot 비중 — PaLI-X에서 robot 50%, PaLM-E에서 robot 66%.

# 왜 비자명한가

LLM 도메인에서 "instruction tuning은 base model 능력을 잃지 않는다"는 것은 일반론. 그러나 robot fine-tuning은:
- Robot data 양이 web 대비 압도적으로 적음 (수십만 episode vs 1B image-text)
- Robot data가 mixture에서 sampling weight를 높여야 함 (자연스러운 비율로는 robot 부족)
- 그럼에도 web 데이터를 완전히 빼면 generalization이 무너짐

즉 "web data는 forgetting 방지용이 아니라, **fine-tuning 단계에서도 web concept을 적극 보강**하는 역할" — 단순 catastrophic forgetting 방지보다 강한 효과.

# 응용 포인트

- **OpenVLA**도 동일 전략 (Open X-Embodiment + web data co-training)
- **π0.5**의 "open-world generalization"도 web data co-training이 핵심 — RT-2 발견을 확장
- Track B 단계에서 SmolVLA / OpenVLA fine-tune 시: **순수 robot data만 쓰지 말고 작은 비중이라도 web data를 같이 흘리는 것을 기본 권장**

# 한계·미해결

- 정확히 어떤 web data가 어느 정도 도움이 되는지 (VQA vs captioning vs OCR)는 분리 안 됨
- Symbol understanding vs math reasoning에서 PaLM-E가 PaLI-X를 일부 task에서 이긴 것은 **각 backbone의 pretraining mixture 차이** 때문 → web pretraining mix design이 직접적으로 emergent capability 결정

# 관련 메모리

- [[rt2-action-as-token-paradigm]]
- [[baseline-papers-scope]]
