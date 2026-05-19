---
name: openvla-dual-vision-encoder-fusion
description: DINOv2(spatial) + SigLIP(semantic) feature를 channel-wise concat하면 manipulation 성능이 단일 인코더 대비 +10pp
metadata:
  type: learning
  created_at: 2026-05-19
  updated_at: 2026-05-19
  source: papers/core-models/OpenVLA-An-Open-Source-Vision-Language-Action-Model.pdf
---

# 한 줄 사실

OpenVLA는 **두 비전 인코더의 patch feature를 channel-wise concatenate**한 후 MLP projector로 LLM embed space에 매핑한다. 단일 SigLIP 대비 BridgeData V2에서 ~10pp, IDEFICS-1 baseline 대비 약 45pp 향상.

# 왜 비자명한가

Robot manipulation은 **semantic("이게 사과")**과 **spatial(정확한 위치·방향·거리)** 두 능력 모두 필요. 그런데 단일 인코더는 한 쪽만 잘함:

| 인코더 | 학습 방식 | 강점 | 약점 |
|---|---|---|---|
| CLIP / SigLIP | Web text-image contrastive | semantic | spatial precision |
| DINOv2 | Self-supervised | spatial/dense | semantic labeling |

**Fusion**은 trivial해 보이지만 LLM/VLM 분야에서는 보통 단일 인코더(CLIP 또는 SigLIP)를 쓴다. OpenVLA가 "Prismatic-7B"(Karamcheti 2024)의 발견을 robot domain에 적용해서 입증.

# 구현 메커니즘

```python
# pseudocode
dinov2_feat = DINOv2(image)      # [B, N_patches, ~1024]
siglip_feat = SigLIP(image)      # [B, N_patches, ~1152]
fused       = torch.cat([dinov2_feat, siglip_feat], dim=-1)  # [B, N, ~2176]
projected   = MLP_2layer(fused)  # [B, N, 4096]  → Llama embed
```

**중요한 디테일**: Patch 단위로 같은 spatial 위치의 feature를 concat. → spatial alignment 유지.

# 정량적 효과 (OpenVLA Section 3.4 ablation)

| Backbone | BridgeData 5-task |
|---|---|
| IDEFICS-1 | baseline |
| LLaVA (CLIP) | +35pp (language grounding 강함) |
| **Prismatic (DINOv2+SigLIP)** | **+10pp on top of LLaVA** |

# 부가 발견: Vision encoder는 freeze하면 안 된다

VLM 학습 표준은 vision encoder freeze (pretrained feature 보존). 그러나 OpenVLA에서는 **fine-tune이 결정적**.

**이유**: VLM benchmark는 거시적 visual concept만 필요하지만, manipulation은 **fine-grained spatial detail** (객체의 mm 단위 위치, 정밀 방향)이 필요. 이는 robot domain 데이터에서 vision encoder를 추가 학습해야 얻을 수 있음.

→ **VLA 학습의 일반 규칙**: vision encoder를 unfreeze.

# 응용 포인트

- 우리 스터디에서 fine-tune 또는 from-scratch VLA 만들 때 **DINOv2 + SigLIP** 또는 동등 spatial+semantic 조합 선택
- Track c new idea: SigLIP 대신 EVA-CLIP, DINOv3 등으로 swap해서 효과 sweep
- Vision encoder freeze vs unfreeze ablation 후속 모델에서도 확인 필요 (SmolVLA, π0 등)

# 한계·미해결

- DINOv2와 SigLIP 사이 weighting (단순 concat이 optimal?)
- 더 큰 vision encoder가 도움이 되는가 (논문은 ~600M에서 멈춤)
- Cross-attention 같은 더 정교한 fusion이 더 좋을 가능성

# 관련 메모리

- [[rt2-action-as-token-paradigm]] — RT-2는 PaLI-X 자체 ViT만 씀. OpenVLA의 fusion이 차별점
- [[openvla-lora-recipe]]
