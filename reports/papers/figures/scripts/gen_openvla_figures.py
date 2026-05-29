"""Generate all SVG figures for OpenVLA.md."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _style import setup, COLORS, save_both, clean_axes

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch

setup()


# =====================================================
# Figure 1: DINOv2 + SigLIP fusion (dual vision encoder)
# =====================================================
def fig_dual_vision_fusion():
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    clean_axes(ax)

    fig.suptitle('OpenVLA: DINOv2 + SigLIP Dual Vision Encoder Fusion',
                 fontsize=14, fontweight='bold', y=0.96)

    # Input image
    img_box = FancyBboxPatch((0.3, 3.2), 2.0, 1.6,
                             boxstyle='round,pad=0.05,rounding_size=0.08',
                             facecolor='#fafafa', edgecolor=COLORS['gray'], linewidth=1.3)
    ax.add_patch(img_box)
    ax.text(1.3, 4.2, 'RGB image\n224×224×3',
            ha='center', va='center', fontsize=10, color=COLORS['text'])
    ax.text(1.3, 3.4, '(1 view)',
            ha='center', va='center', fontsize=9, color=COLORS['text_muted'],
            style='italic')

    # Split arrow to two encoders
    ax.add_patch(FancyArrowPatch((2.3, 4.5), (3.8, 5.6),
                                 arrowstyle='-|>', mutation_scale=14,
                                 color=COLORS['gray'], linewidth=1.4))
    ax.add_patch(FancyArrowPatch((2.3, 3.5), (3.8, 2.6),
                                 arrowstyle='-|>', mutation_scale=14,
                                 color=COLORS['gray'], linewidth=1.4))

    # DINOv2 encoder (top)
    dino_box = FancyBboxPatch((3.9, 5.0), 2.6, 1.6,
                              boxstyle='round,pad=0.05,rounding_size=0.08',
                              facecolor=COLORS['purple_light'], alpha=0.55,
                              edgecolor=COLORS['purple'], linewidth=1.8)
    ax.add_patch(dino_box)
    ax.text(5.2, 6.2, 'DINOv2', fontsize=12, fontweight='bold',
            ha='center', color=COLORS['purple'])
    ax.text(5.2, 5.6, 'self-supervised\n→ spatial features',
            ha='center', fontsize=9, color=COLORS['text'], style='italic')

    # SigLIP encoder (bottom)
    sig_box = FancyBboxPatch((3.9, 1.5), 2.6, 1.6,
                             boxstyle='round,pad=0.05,rounding_size=0.08',
                             facecolor=COLORS['green_light'], alpha=0.6,
                             edgecolor=COLORS['green'], linewidth=1.8)
    ax.add_patch(sig_box)
    ax.text(5.2, 2.7, 'SigLIP', fontsize=12, fontweight='bold',
            ha='center', color=COLORS['green'])
    ax.text(5.2, 2.1, 'image-text contrastive\n→ semantic features',
            ha='center', fontsize=9, color=COLORS['text'], style='italic')

    # Features arrows to concat
    ax.add_patch(FancyArrowPatch((6.55, 5.8), (8.0, 4.6),
                                 arrowstyle='-|>', mutation_scale=14,
                                 color=COLORS['purple'], linewidth=1.5))
    ax.add_patch(FancyArrowPatch((6.55, 2.3), (8.0, 3.5),
                                 arrowstyle='-|>', mutation_scale=14,
                                 color=COLORS['green'], linewidth=1.5))

    # Concat box
    concat_box = FancyBboxPatch((8.1, 3.3), 2.4, 1.5,
                                boxstyle='round,pad=0.05,rounding_size=0.08',
                                facecolor='#fff4e0', edgecolor=COLORS['orange'], linewidth=1.5)
    ax.add_patch(concat_box)
    ax.text(9.3, 4.4, 'Channel-wise\nCONCAT',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color=COLORS['orange'])
    ax.text(9.3, 3.6, '[N, d_dino+d_siglip]',
            ha='center', va='center', fontsize=9, color=COLORS['text'],
            family='monospace')

    # MLP projector
    ax.add_patch(FancyArrowPatch((10.55, 4.05), (11.6, 4.05),
                                 arrowstyle='-|>', mutation_scale=14,
                                 color=COLORS['gray'], linewidth=1.4))
    proj_box = FancyBboxPatch((11.7, 3.5), 2.0, 1.1,
                              boxstyle='round,pad=0.05,rounding_size=0.08',
                              facecolor=COLORS['gray_light'],
                              edgecolor=COLORS['gray'], linewidth=1.3)
    ax.add_patch(proj_box)
    ax.text(12.7, 4.2, '2-layer MLP\nProjector',
            ha='center', va='center', fontsize=10, color=COLORS['text'])
    ax.text(12.7, 3.65, '→ Llama-2 dim',
            ha='center', va='center', fontsize=9, color=COLORS['text_muted'])

    # Annotation: why dual?
    annot = FancyBboxPatch((0.3, 0.2), 13.4, 1.0,
                           boxstyle='round,pad=0.03,rounding_size=0.05',
                           facecolor='#fff8e7', edgecolor='#d4a05a', linewidth=1.0)
    ax.add_patch(annot)
    ax.text(7.0, 0.85, '왜 두 encoder를 합치는가?',
            ha='center', fontsize=11, fontweight='bold', color='#8b6914')
    ax.text(7.0, 0.45,
            'Robot manipulation에는 semantic (객체 인식)과 spatial (정확한 위치) '
            '모두 필요. 단일 encoder는 한쪽만 강함.',
            ha='center', fontsize=10, color=COLORS['text'])

    plt.subplots_adjust(top=0.91, bottom=0.04, left=0.02, right=0.98)
    save_both(fig, 'openvla_dual_vision_fusion')
    plt.close(fig)


# =====================================================
# Figure 2: LoRA fine-tuning comparison (bar chart + memory)
# =====================================================
def fig_lora_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor('white')
    fig.suptitle('OpenVLA Fine-tuning Strategies: Performance vs Cost',
                 fontsize=14, fontweight='bold', y=0.98)

    strategies = ['Full FT', 'Last layer\nonly', 'Frozen\nvision',
                  'Sandwich', 'LoRA\nr=32', 'LoRA\nr=64']
    success = [69.7, 30.3, 47.0, 62.1, 68.2, 68.2]
    train_params = [7188, 465, 6760, 914, 97.6, 195.2]
    vram = [163, 51, 156, 64, 60, 60]
    colors = [COLORS['red_light'], COLORS['gray_light'], COLORS['orange_light'],
              '#f5d4a0', COLORS['green_light'], COLORS['green_light']]
    edge_colors = [COLORS['red'], COLORS['gray'], COLORS['orange'],
                   '#d4a05a', COLORS['green'], COLORS['green']]

    # Left: success rate
    x = np.arange(len(strategies))
    bars = ax1.bar(x, success, color=colors, edgecolor=edge_colors, linewidth=1.5)
    for bar, val in zip(bars, success):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                 f'{val:.1f}%', ha='center', fontsize=10, color=COLORS['text'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=9)
    ax1.set_ylabel('Success Rate (%)', fontsize=11)
    ax1.set_ylim(0, 85)
    ax1.set_title('Performance (Franka-Tabletop)',
                  fontsize=11, color=COLORS['text'], pad=10)
    ax1.axhline(y=69.7, color=COLORS['red'], linestyle='--', alpha=0.5, linewidth=1.2)
    ax1.text(5.5, 71.5, 'Full FT baseline',
             ha='right', fontsize=9, color=COLORS['red'], style='italic')
    clean_axes(ax1, keep=('bottom', 'left'))
    ax1.spines['bottom'].set_color(COLORS['gray'])
    ax1.spines['left'].set_color(COLORS['gray'])

    # Right: train params (log scale) + vram
    ax2_twin = ax2.twinx()
    bars2 = ax2.bar(x - 0.2, train_params, width=0.4,
                    color=colors, edgecolor=edge_colors, linewidth=1.5,
                    label='Train params (M)')
    bars3 = ax2_twin.bar(x + 0.2, vram, width=0.4,
                         color=[COLORS['purple_light']] * len(x),
                         edgecolor=COLORS['purple'], linewidth=1.5,
                         alpha=0.7, label='VRAM (GB)')

    ax2.set_yscale('log')
    ax2.set_ylabel('Trainable params (M, log scale)', fontsize=10,
                   color=COLORS['text'])
    ax2_twin.set_ylabel('VRAM (GB)', fontsize=10, color=COLORS['purple'])
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, fontsize=9)
    ax2.set_title('Compute Cost (params + memory)',
                  fontsize=11, color=COLORS['text'], pad=10)

    # Annotations on params
    for i, (bar, val) in enumerate(zip(bars2, train_params)):
        ax2.text(bar.get_x() + bar.get_width() / 2, val * 1.4,
                 f'{val:.0f}M', ha='center', fontsize=8, color=COLORS['text'])
    for bar, val in zip(bars3, vram):
        ax2_twin.text(bar.get_x() + bar.get_width() / 2, val + 4,
                      f'{val}GB', ha='center', fontsize=8,
                      color=COLORS['purple'])

    clean_axes(ax2, keep=('bottom', 'left'))
    clean_axes(ax2_twin, keep=('bottom', 'right'))

    # Sweet spot annotation
    ax1.annotate('Sweet spot ★\n(1.4% params, single A100)',
                 xy=(4, 68.2), xytext=(2.7, 80),
                 fontsize=10, color=COLORS['green'], fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=1.5),
                 ha='center')

    plt.subplots_adjust(top=0.91, bottom=0.12, left=0.08, right=0.92, wspace=0.3)
    save_both(fig, 'openvla_lora_comparison')
    plt.close(fig)


# =====================================================
# Figure 3: Quantile-based discretization (vs min-max)
# =====================================================
def fig_quantile_discretization():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor('white')
    fig.suptitle('Quantile-based vs Min-max Action Discretization',
                 fontsize=14, fontweight='bold', y=0.98)

    # Generate sample distribution with outliers
    np.random.seed(42)
    main = np.random.normal(0, 0.3, 950)
    outliers = np.random.uniform(-5, 5, 50)  # ~5% outliers far away
    data = np.concatenate([main, outliers])
    data = np.clip(data, -6, 6)

    # ============ Left: min-max ============
    ax1.set_title('RT-2 / 기존: min-max discretization',
                  fontsize=11, color=COLORS['red'], pad=10)
    ax1.hist(data, bins=80, color=COLORS['gray_light'], edgecolor=COLORS['gray'])

    a_min, a_max = data.min(), data.max()
    bins_minmax = np.linspace(a_min, a_max, 257)
    bin_width_mm = (a_max - a_min) / 256

    ax1.axvline(a_min, color=COLORS['red'], linewidth=2.0, linestyle='--', label='min')
    ax1.axvline(a_max, color=COLORS['red'], linewidth=2.0, linestyle='--', label='max')

    # Highlight effective region (where most data lives)
    q01, q99 = np.quantile(data, [0.01, 0.99])
    ax1.axvspan(q01, q99, alpha=0.2, color=COLORS['orange'])
    ax1.text(0, ax1.get_ylim()[1] * 0.7,
             f'정상 분포\n[{q01:.2f}, {q99:.2f}]',
             ha='center', fontsize=10, color=COLORS['orange'], fontweight='bold')

    # Bin width annotation
    ax1.text(0, ax1.get_ylim()[1] * 0.92,
             f'전체 range = {a_max - a_min:.1f}  →  bin width = {bin_width_mm:.3f}',
             ha='center', fontsize=10, color=COLORS['text'])
    ax1.text(0, -ax1.get_ylim()[1] * 0.08,
             '문제: outlier가 range 부풀림 → 정상 분포가 256 bin 중 적은 일부만 차지',
             ha='center', fontsize=9, color=COLORS['red'], style='italic',
             transform=ax1.transData)

    ax1.set_xlim(-6, 6)
    ax1.set_xlabel('action value', fontsize=10)
    ax1.set_ylabel('frequency', fontsize=10)
    clean_axes(ax1, keep=('bottom', 'left'))

    # ============ Right: quantile-based ============
    ax2.set_title('OpenVLA: 1-99 percentile discretization',
                  fontsize=11, color=COLORS['green'], pad=10)
    ax2.hist(data, bins=80, color=COLORS['gray_light'], edgecolor=COLORS['gray'])

    bins_q = np.linspace(q01, q99, 257)
    bin_width_q = (q99 - q01) / 256

    ax2.axvline(q01, color=COLORS['green'], linewidth=2.0, linestyle='--')
    ax2.axvline(q99, color=COLORS['green'], linewidth=2.0, linestyle='--')

    ax2.axvspan(q01, q99, alpha=0.2, color=COLORS['green'])
    ax2.text(0, ax2.get_ylim()[1] * 0.7,
             f'1-99 percentile\n[{q01:.2f}, {q99:.2f}]',
             ha='center', fontsize=10, color=COLORS['green'], fontweight='bold')

    ax2.text(0, ax2.get_ylim()[1] * 0.92,
             f'effective range = {q99 - q01:.2f}  →  bin width = {bin_width_q:.4f}',
             ha='center', fontsize=10, color=COLORS['text'])

    # Calculate improvement
    improvement = bin_width_mm / bin_width_q
    ax2.text(0, -ax2.get_ylim()[1] * 0.08,
             f'개선: outlier 1% 제거 → bin width ~{improvement:.0f}배 정밀',
             ha='center', fontsize=9, color=COLORS['green'], style='italic',
             transform=ax2.transData)

    ax2.set_xlim(-6, 6)
    ax2.set_xlabel('action value', fontsize=10)
    ax2.set_ylabel('frequency', fontsize=10)
    clean_axes(ax2, keep=('bottom', 'left'))

    plt.subplots_adjust(top=0.90, bottom=0.15, left=0.07, right=0.97, wspace=0.18)
    save_both(fig, 'openvla_quantile_discretization')
    plt.close(fig)


if __name__ == '__main__':
    fig_dual_vision_fusion()
    fig_lora_comparison()
    fig_quantile_discretization()
    print('All OpenVLA figures generated.')
