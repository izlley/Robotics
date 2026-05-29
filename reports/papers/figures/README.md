# reports/papers/figures/

Self-contained SVG diagrams for each paper summary. All figures are programmatically generated with matplotlib for reproducibility.

## 디렉토리 구조

```
figures/
├── README.md                    # 본 파일
├── scripts/                     # 생성 스크립트
│   ├── _style.py                # 공통 style helper (color palette, fonts)
│   ├── gen_rt2_figures.py       # RT-2 figures
│   ├── gen_openvla_figures.py   # OpenVLA figures
│   ├── gen_smolvla_figures.py   # SmolVLA figures
│   ├── gen_pi0_figures.py       # π0 figures
│   └── gen_pi05_figures.py      # π0.5 figures
├── *.svg                        # vector format (preferred for embedding)
└── *.png                        # raster preview (150 dpi)
```

## 파일 명명 규칙

`<paper>_<concept>.svg` 형태:

- `rt2_*.svg` — RT-2 figures
- `openvla_*.svg` — OpenVLA figures
- `smolvla_*.svg` — SmolVLA figures
- `pi0_*.svg` — π0 figures
- `pi05_*.svg` — π0.5 figures

## 현재 figure list (17개)

| Paper | Figure | Concept |
|---|---|---|
| RT-2 | `rt2_architecture_flow` | 전체 VLM-as-VLA 흐름 |
| RT-2 | `rt2_action_tokenization` | 256 bin discretize → text token |
| RT-2 | `rt2_co_finetuning_ablation` | scratch/FT/co-FT 비교 |
| OpenVLA | `openvla_dual_vision_fusion` | DINOv2 + SigLIP channel-wise concat |
| OpenVLA | `openvla_lora_comparison` | Fine-tuning strategies (Full FT, LoRA, etc.) |
| OpenVLA | `openvla_quantile_discretization` | Min-max vs 1-99 percentile |
| SmolVLA | `smolvla_layer_skipping` | VLM 16층 중 앞 8층만 사용 |
| SmolVLA | `smolvla_pixel_shuffle` | 공간→채널 정보 이동 |
| SmolVLA | `smolvla_action_expert_blocks` | CA+SA interleaved blocks |
| SmolVLA | `smolvla_async_inference` | Sync vs Async timeline |
| π0 | `pi0_moe_architecture` | 단일 transformer 안의 두 weight set |
| π0 | `pi0_attention_mask` | Block-wise causal attention |
| π0 | `pi0_tau_distribution` | Shifted Beta(1.5,1) τ sampling |
| π0 | `pi0_linear_path` | Noise → action linear path (velocity) |
| π0.5 | `pi05_hybrid_fast_flow` | FAST pre-train + Flow post-train |
| π0.5 | `pi05_tau_injection_comparison` | π0 MLP fusion vs π0.5 AdaLN |
| π0.5 | `pi05_hierarchical_inference` | Same model, two-stage inference |
| π0.5 | `pi05_data_mixture` | 5+ source heterogeneous co-training |

## 생성 방법

```bash
cd /workspace/izlley/sllm/Robotics/reports/papers/figures/scripts
python3 gen_<paper>_figures.py
```

`_style.py`의 `setup()`이 NanumGothic 한글 폰트를 로드하고 공통 color palette를 설정. 모든 figure는 SVG + PNG 둘 다 저장.

## 의존성

```bash
pip install matplotlib scipy numpy pypdf
apt-get install fonts-nanum fonts-noto-cjk  # Korean fonts
```

## Markdown 임베딩 규칙

```markdown
![<concise alt>](figures/<paper>_<concept>.svg)
*<italic caption: 무엇을 보여주는지 1-2줄로>*
```

SVG가 GitHub에서 직접 렌더링되므로 PNG보다 SVG 사용 우선. PNG는 backup용 + 환경에 따른 fallback.

## 스타일 규칙

- Color palette: `_style.COLORS` 사용 (consistent across papers)
  - 보라 (purple): VLM, perception
  - 초록 (green): action, success
  - 주황 (orange): special highlights (peak, sweet spot)
  - 빨강 (red): cutoff, failure, blocked
  - 회색 (gray): input/output, neutral
- ASCII art는 box 안에 한글 X (NanumGothic이 일부 unicode를 못 그림)
- minus 기호는 ASCII `-` 또는 `−` (NanumGothic 호환) 사용

## 향후 추가 figure 후보

- π★0.6: RECAP RL pipeline, Q-value visualization
- π0.7: Memory module, steerable demo
- GR00T N1: Humanoid balance + manipulation hierarchy
- Synthesis: Architecture evolution tree, paradigm comparison

각 paper 정독 시 figure 생성 script도 함께 작성하여 재현성 보장.
