"""Generate all SVG figures for RT-2.md."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _style import setup, COLORS, save_both, clean_axes

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch

setup()


# =====================================================
# Figure 1: Action tokenization (256 bin scheme)
# =====================================================
def fig_action_tokenization():
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.set_xticks([])
    ax.set_yticks([])
    clean_axes(ax)

    fig.suptitle('RT-2 Action Tokenization: continuous → 256 bin → text token',
                 fontsize=14, fontweight='bold', y=0.97)

    # Step 1: Continuous action vector
    cont_box = FancyBboxPatch((0.3, 6.5), 4.5, 1.5,
                              boxstyle='round,pad=0.05,rounding_size=0.08',
                              facecolor='#fafafa', edgecolor=COLORS['gray'], linewidth=1.3)
    ax.add_patch(cont_box)
    ax.text(2.55, 7.6, 'Step 1: Continuous action', fontsize=11, fontweight='bold',
            ha='center', color=COLORS['text'])
    ax.text(2.55, 7.0, '[terminate, Δpos_x, Δpos_y, Δpos_z,\n  Δrot_x, Δrot_y, Δrot_z, gripper]',
            ha='center', va='center', fontsize=9, color=COLORS['text'],
            family='monospace')

    # Arrow down
    ax.add_patch(FancyArrowPatch((2.55, 6.45), (2.55, 5.65),
                                 arrowstyle='-|>', mutation_scale=16,
                                 color=COLORS['gray'], linewidth=1.5))
    ax.text(2.55, 6.1, 'discretize\neach dim',
            ha='center', va='center', fontsize=9, color=COLORS['text_muted'],
            style='italic')

    # Step 2: Discretize to 256 bins
    disc_box = FancyBboxPatch((0.3, 4.0), 4.5, 1.5,
                              boxstyle='round,pad=0.05,rounding_size=0.08',
                              facecolor=COLORS['purple_light'], alpha=0.5,
                              edgecolor=COLORS['purple'], linewidth=1.5)
    ax.add_patch(disc_box)
    ax.text(2.55, 5.1, 'Step 2: Bin index (0~255)',
            fontsize=11, fontweight='bold', ha='center', color=COLORS['purple'])
    ax.text(2.55, 4.5, '[1, 128, 91, 241,\n  5, 101, 127, 142]',
            ha='center', va='center', fontsize=10, color=COLORS['text'],
            family='monospace')

    # Right side: bin width example
    ax.text(5.2, 6.0,
            'bin(a) = ⌊ (a − a_min) / (a_max − a_min) × 256 ⌋',
            ha='left', fontsize=10, color=COLORS['text'], family='monospace')
    ax.text(5.2, 5.55,
            '예: Δpos ∈ [−0.05, +0.05] m, 256 bin',
            ha='left', fontsize=10, color=COLORS['text'])
    ax.text(5.2, 5.1,
            'bin width = 0.1m / 256 ≈ 0.39mm',
            ha='left', fontsize=10, color=COLORS['orange'], fontweight='bold')

    # Arrow down
    ax.add_patch(FancyArrowPatch((2.55, 3.95), (2.55, 3.15),
                                 arrowstyle='-|>', mutation_scale=16,
                                 color=COLORS['gray'], linewidth=1.5))
    ax.text(2.55, 3.55, 'token slot\nmapping',
            ha='center', va='center', fontsize=9, color=COLORS['text_muted'],
            style='italic')

    # Step 3: Map to VLM text tokens
    text_box = FancyBboxPatch((0.3, 1.5), 4.5, 1.5,
                              boxstyle='round,pad=0.05,rounding_size=0.08',
                              facecolor=COLORS['green_light'], alpha=0.6,
                              edgecolor=COLORS['green'], linewidth=1.5)
    ax.add_patch(text_box)
    ax.text(2.55, 2.6, 'Step 3: Text token string',
            fontsize=11, fontweight='bold', ha='center', color=COLORS['green'])
    ax.text(2.55, 2.0, '"1 128 91 241 5 101 127 142"',
            ha='center', va='center', fontsize=10, color=COLORS['text'],
            family='monospace')

    # Right side: two mapping schemes
    map_box = FancyBboxPatch((5.2, 0.3), 8.5, 4.2,
                             boxstyle='round,pad=0.05,rounding_size=0.08',
                             facecolor='#fff8e7', edgecolor='#d4a05a', linewidth=1.2)
    ax.add_patch(map_box)
    ax.text(9.45, 4.2, 'Token slot mapping (VLM-specific)',
            ha='center', fontsize=11, fontweight='bold', color='#8b6914')

    # PaLI-X
    ax.text(5.5, 3.5, 'PaLI-X:', fontsize=10, fontweight='bold', color=COLORS['text'])
    ax.text(6.2, 3.5, 'Tokenizer에 정수 0~999 unique token이 있음',
            fontsize=10, color=COLORS['text'])
    ax.text(6.2, 3.05, '→ bin index를 그대로 token "0"~"255"로 사용',
            fontsize=10, color=COLORS['text'])

    # PaLM-E (symbol tuning)
    ax.text(5.5, 2.3, 'PaLM-E:', fontsize=10, fontweight='bold', color=COLORS['text'])
    ax.text(6.4, 2.3, 'Tokenizer에 그런 편의 없음',
            fontsize=10, color=COLORS['text'])
    ax.text(6.4, 1.85, '→ Symbol tuning: vocab에서 가장 안 쓰이는',
            fontsize=10, color=COLORS['text'])
    ax.text(6.4, 1.45, '   256개 token을 overwrite하여 action으로 재학습',
            fontsize=10, color=COLORS['text'])

    ax.text(9.45, 0.7,
            '결과: VLM의 다음 token 예측 그대로 → action 출력',
            ha='center', fontsize=10, color='#8b6914', fontweight='bold',
            style='italic')

    plt.subplots_adjust(top=0.92, bottom=0.04, left=0.02, right=0.98)
    save_both(fig, 'rt2_action_tokenization')
    plt.close(fig)


# =====================================================
# Figure 2: Co-fine-tuning vs Fine-tuning vs Scratch
# =====================================================
def fig_co_finetuning_ablation():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor('white')

    fig.suptitle('RT-2 Co-fine-tuning Ablation (Unseen Generalization)',
                 fontsize=14, fontweight='bold', y=0.96)

    methods = ['from scratch\n(no VLM PT)', 'fine-tuning only\n(robot data only)',
               'co-fine-tuning\n(robot + web data)']
    perf_5b = [9, 42, 44]
    perf_55b = [None, 52, 63]  # from scratch skipped for 55B
    colors_m = [COLORS['red_light'], COLORS['orange_light'], COLORS['green_light']]
    edges_m = [COLORS['red'], COLORS['orange'], COLORS['green']]

    x = np.arange(len(methods))
    width = 0.38

    # 5B bars
    bars_5b = ax.bar(x - width / 2, perf_5b, width,
                     color=colors_m, edgecolor=edges_m, linewidth=1.5,
                     label='RT-2-PaLI-X 5B')

    # 55B bars (skip None)
    perf_55b_clean = [v if v is not None else 0 for v in perf_55b]
    bars_55b = ax.bar(x + width / 2, perf_55b_clean, width,
                      color=colors_m, edgecolor=edges_m, linewidth=1.5,
                      hatch='///', alpha=0.7, label='RT-2-PaLI-X 55B')

    # Annotations
    for bar, val in zip(bars_5b, perf_5b):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                f'{val}%', ha='center', fontsize=11, color=COLORS['text'])
    for bar, val in zip(bars_55b, perf_55b):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                    f'{val}%', ha='center', fontsize=11, color=COLORS['text'])
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 2,
                    '(skipped)', ha='center', fontsize=9, color=COLORS['gray'],
                    style='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel('Unseen Generalization Avg (%)', fontsize=11)
    ax.set_ylim(0, 75)
    clean_axes(ax, keep=('bottom', 'left'))
    ax.spines['bottom'].set_color(COLORS['gray'])
    ax.spines['left'].set_color(COLORS['gray'])
    ax.legend(loc='upper left', fontsize=10, frameon=False)

    # Key insights as annotations
    ax.annotate('', xy=(1, 52), xytext=(0, 9),
                arrowprops=dict(arrowstyle='->', color=COLORS['purple'], lw=1.5))
    ax.text(0.5, 32, '+33pp\n(VLM pre-training)',
            ha='center', fontsize=10, color=COLORS['purple'], fontweight='bold')

    ax.annotate('', xy=(2 + width / 2, 63), xytext=(1 + width / 2, 52),
                arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=1.5))
    ax.text(1.5 + width / 2, 60.5, '+11pp\n(co-FT vs FT)',
            ha='center', fontsize=10, color=COLORS['green'], fontweight='bold')

    # Bottom takeaway
    fig.text(0.5, 0.04,
             '핵심: (1) VLM pre-training이 결정적,  (2) Co-fine-tuning이 FT보다 우수,  '
             '(3) 모델 클수록 차이 증폭',
             ha='center', fontsize=11, color=COLORS['text'], fontweight='bold')

    plt.subplots_adjust(top=0.90, bottom=0.11, left=0.08, right=0.95)
    save_both(fig, 'rt2_co_finetuning_ablation')
    plt.close(fig)


# =====================================================
# Figure 3: Architecture flow (VLM + action token output)
# =====================================================
def fig_architecture_flow():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.set_xticks([])
    ax.set_yticks([])
    clean_axes(ax)

    fig.suptitle('RT-2 Architecture: VLM as single end-to-end VLA',
                 fontsize=14, fontweight='bold', y=0.96)

    # Inputs: image and text
    img_box = FancyBboxPatch((0.3, 4.5), 2.2, 1.4,
                             boxstyle='round,pad=0.05,rounding_size=0.08',
                             facecolor='#fafafa', edgecolor=COLORS['gray'])
    ax.add_patch(img_box)
    ax.text(1.4, 5.4, 'Image', fontsize=11, fontweight='bold',
            ha='center', color=COLORS['text'])
    ax.text(1.4, 4.85, 'RGB camera',
            ha='center', fontsize=9, color=COLORS['text_muted'])

    text_box = FancyBboxPatch((0.3, 2.5), 2.2, 1.4,
                              boxstyle='round,pad=0.05,rounding_size=0.08',
                              facecolor='#fafafa', edgecolor=COLORS['gray'])
    ax.add_patch(text_box)
    ax.text(1.4, 3.4, 'Language', fontsize=11, fontweight='bold',
            ha='center', color=COLORS['text'])
    ax.text(1.4, 2.85, '"Q: what action ...\nfor [task]? A:"',
            ha='center', fontsize=8, color=COLORS['text_muted'],
            style='italic')

    # Arrows to VLM
    ax.add_patch(FancyArrowPatch((2.55, 5.2), (4.2, 4.6),
                                 arrowstyle='-|>', mutation_scale=14,
                                 color=COLORS['gray'], linewidth=1.4))
    ax.add_patch(FancyArrowPatch((2.55, 3.2), (4.2, 3.8),
                                 arrowstyle='-|>', mutation_scale=14,
                                 color=COLORS['gray'], linewidth=1.4))

    # VLM box (massive)
    vlm_box = FancyBboxPatch((4.3, 2.3), 5.0, 3.5,
                             boxstyle='round,pad=0.06,rounding_size=0.1',
                             facecolor=COLORS['purple_light'], alpha=0.5,
                             edgecolor=COLORS['purple'], linewidth=2.0)
    ax.add_patch(vlm_box)
    ax.text(6.8, 5.4, 'VLM Backbone',
            ha='center', fontsize=13, fontweight='bold', color=COLORS['purple'])
    ax.text(6.8, 4.8, 'PaLI-X 5B/55B  or  PaLM-E 12B',
            ha='center', fontsize=11, color=COLORS['text'], style='italic')
    ax.text(6.8, 4.0,
            '• 새 layer/parameter 추가 없음\n'
            '• next-token prediction loss 그대로\n'
            '• action token = 일반 text token',
            ha='center', va='center', fontsize=10, color=COLORS['text'])

    # Co-fine-tuning indicator
    ax.text(6.8, 2.6,
            'Co-fine-tuning: web data 50-66% + robot data',
            ha='center', fontsize=9, color=COLORS['orange'],
            fontweight='bold')

    # Arrow to output
    ax.add_patch(FancyArrowPatch((9.35, 4.05), (10.5, 4.05),
                                 arrowstyle='-|>', mutation_scale=18,
                                 color=COLORS['gray'], linewidth=1.5))

    # Output box
    out_box = FancyBboxPatch((10.55, 2.7), 3.2, 2.7,
                             boxstyle='round,pad=0.06,rounding_size=0.1',
                             facecolor=COLORS['green_light'], alpha=0.6,
                             edgecolor=COLORS['green'], linewidth=1.8)
    ax.add_patch(out_box)
    ax.text(12.15, 5.0, 'Action token output',
            ha='center', fontsize=11, fontweight='bold', color=COLORS['green'])
    ax.text(12.15, 4.3, '"1 128 91 241\n5 101 127 142"',
            ha='center', va='center', fontsize=10, color=COLORS['text'],
            family='monospace')
    ax.text(12.15, 3.2,
            '↓ de-tokenize\n6-DoF Cartesian\n+ gripper command',
            ha='center', va='center', fontsize=9, color=COLORS['text_muted'])

    # Bottom: output constraint
    fig.text(0.5, 0.08,
             'Decoding 시 output constraint: robot prompt가 들어왔을 때만 action vocab(256개)으로 sampling 제한',
             ha='center', fontsize=10, color=COLORS['text'])
    fig.text(0.5, 0.025,
             '일반 vision-language task에서는 전체 vocab 그대로 (multi-task multi-output)',
             ha='center', fontsize=9, color=COLORS['text_muted'], style='italic')

    plt.subplots_adjust(top=0.92, bottom=0.16, left=0.02, right=0.98)
    save_both(fig, 'rt2_architecture_flow')
    plt.close(fig)


if __name__ == '__main__':
    fig_action_tokenization()
    fig_co_finetuning_ablation()
    fig_architecture_flow()
    print('All RT-2 figures generated.')
