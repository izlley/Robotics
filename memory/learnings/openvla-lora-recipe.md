---
name: openvla-lora-recipe
description: VLA fine-tuning에서 LoRA rank=32가 full FT와 동등 성능을 1.4% params로 달성 — LLM의 LoRA가 VLA에도 그대로 통함
metadata:
  type: learning
  created_at: 2026-05-19
  updated_at: 2026-05-19
  source: papers/core-models/OpenVLA-An-Open-Source-Vision-Language-Action-Model.pdf
---

# 한 줄 사실

OpenVLA 7B를 새 robot task에 fine-tune할 때, **all linear layer에 LoRA rank=32**를 적용하면 full fine-tuning(7188M params)과 동등한 68.2% 성공률을 **97.6M params(1.4%)**, **single A100 80GB**, **10-15시간**으로 달성. Full FT는 2x A100 + 5-15시간 → **8x compute 절감**.

# 정량 데이터 (OpenVLA Table 1)

| Strategy | Success | Train Params | VRAM |
|---|---|---|---|
| Full FT | 69.7% | 7188M (100%) | 163 GB (2x A100) |
| Last layer only | 30.3% | 465M (6.5%) | 51 GB |
| Frozen vision | 47.0% | 6760M (94%) | 156 GB |
| Sandwich (vision+emb+last) | 62.1% | 914M (12.7%) | 64 GB |
| **LoRA rank=32** | **68.2%** | **97.6M (1.4%)** | **60 GB** ★ |
| LoRA rank=64 | 68.2% | 195.2M (2.7%) | 60 GB |

# 왜 비자명한가 (LLM에서는 당연한데)

LLM 분야에서 LoRA가 잘 작동하는 건 이미 검증됨. 그러나 VLA는 다음 이유로 작동 안 할 가능성도 있었음:

1. **Vision encoder도 학습해야 한다** (위의 frozen vision 결과가 보여줌) → LoRA가 ViT 위에서도 작동하나?
2. **Action prediction은 새로운 modality** → low intrinsic rank 가설이 깨질 수 있음
3. **Robot data는 narrow하고 specific** → intrinsic update가 크지 않을까?

OpenVLA가 정량 검증: **all linear layer (attention + MLP, 비전 인코더 포함)에 LoRA 적용하면 작동**. 즉:
- Attention K/Q/V/O LoRA
- MLP linear LoRA
- ViT linear LoRA도 포함
- → VLA fine-tuning도 본질적으로 **low intrinsic rank task**

# 핵심 디테일

- **Rank**: r=32 vs r=64 차이 없음 → r=32 권장 (parameter ↓)
- **Target**: **all linear layers** (LoRA 전형보다 광범위)
- **No need for**: warmup, custom schedule. LR 2e-5 그대로

# 비교 — 다른 PEFT 전략은 안 됨

| Strategy | 왜 작동 안 함 |
|---|---|
| Last layer only | Visual feature가 robot domain으로 안 옴 → 위치 정밀도 ↓ |
| Frozen vision | 위 + 메모리 절감 미미 (94% 학습) |
| Sandwich (vision+emb+last) | LoRA에 비해 7pp 낮음, 비슷한 메모리 |

→ **LoRA만이 sweet spot**: memory ↓, params ↓, performance == full FT

# 응용 포인트

**우리 스터디 Track B 진입 시 실용 recipe**:

```python
from peft import LoraConfig, get_peft_model
config = LoraConfig(
    r=32, lora_alpha=64,
    target_modules="all-linear",  # ViT + projector + Llama 모두
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(openvla_model, config)
# → single A100, 10-150 demos, 10-15h
```

**Quantization 결합 가능성** (논문 미검증):
- QLoRA (4-bit weight + LoRA adapter): 추가 메모리 절감 가능
- 우리 스터디에서 시도 가능한 new idea

# 한계·미해결

- Multi-task fine-tune (한 모델로 여러 새 task)에서 LoRA가 어떻게 작동하는지 검증 안 됨
- LoRA의 task-specific adapter swap이 robot domain에서도 가능한지 (LLM에서는 OK)
- 더 정교한 PEFT (DoRA, AdaLoRA 등)와 비교 안 됨
- Vision encoder의 어느 layer가 가장 중요한지 (LoRA target layer sweep)

# LLM 도구와의 1:1 대응

| LLM PEFT 도구 | OpenVLA에서의 활용 가능성 |
|---|---|
| LoRA (r=32) | ✅ 검증됨 |
| QLoRA (4-bit + LoRA) | 검증 안 됨, 가능성 ↑ |
| DoRA | 검증 안 됨 |
| AdaLoRA | 검증 안 됨 |
| Prefix tuning | 검증 안 됨 |
| Adapter (Houlsby) | 검증 안 됨 |

→ OpenVLA가 길을 열었고, **나머지 PEFT 기법들이 VLA로 transfer 가능한지가 흥미로운 new idea 후보**.

# 관련 메모리

- [[openvla-dual-vision-encoder-fusion]]
- [[rt2-co-fine-tuning-trick]] — 다른 종류의 efficient training trick (web co-training)
