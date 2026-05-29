"""Generate all SVG figures for SmolVLA.md."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _style import setup, COLORS, save_both, clean_axes

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch

setup()


# =====================================================
# Figure 1: Action Expert per-block structure (CA + SA interleaved)
# =====================================================
def fig_action_expert_blocks():
    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.set_xticks([])
    ax.set_yticks([])
    clean_axes(ax)

    fig.suptitle('SmolVLA Action Expert: Interleaved CA + SA blocks',
                 fontsize=14, fontweight='bold', y=0.97)
    ax.set_title('CA (Cross-Attention with VLM features) → SA (causal Self-Attention within action chunk) → repeat',
                 fontsize=11, color=COLORS['text_muted'], pad=8)

    # VLM features (left, vertical)
    vlm_box = FancyBboxPatch((0.3, 2.5), 2.0, 4.0,
                             boxstyle='round,pad=0.05,rounding_size=0.1',
                             facecolor=COLORS['purple_light'], alpha=0.5,
                             edgecolor=COLORS['purple'], linewidth=1.8)
    ax.add_patch(vlm_box)
    ax.text(1.3, 6.1, 'VLM features', ha='center', fontsize=12, fontweight='bold',
            color=COLORS['purple'])
    ax.text(1.3, 5.6, '(SmolLM-2 layer 8\noutput, frozen)',
            ha='center', fontsize=9, color=COLORS['text'], style='italic')
    ax.text(1.3, 4.6, '[S=149,\n  d_a=432]',
            ha='center', fontsize=10, color=COLORS['text'], family='monospace')
    ax.text(1.3, 3.4, 'K, V source\nfor all CA',
            ha='center', fontsize=9, color=COLORS['text_muted'])

    # Action token input (bottom)
    action_in = FancyBboxPatch((4.0, 0.3), 7.0, 0.9,
                               boxstyle='round,pad=0.05,rounding_size=0.06',
                               facecolor=COLORS['green_light'], alpha=0.6,
                               edgecolor=COLORS['green'], linewidth=1.6)
    ax.add_patch(action_in)
    ax.text(7.5, 0.75, 'Noisy action chunk A$^τ$  [n=50, action_dim=7]',
            ha='center', va='center', fontsize=11, color=COLORS['text'])

    # Block stack (alternating CA / SA)
    block_specs = [
        ('Block 1: Cross-Attention', 'CA', 'Q ← action,  K,V ← VLM features',
         COLORS['purple_light'], COLORS['purple']),
        ('Block 2: Self-Attention',  'SA', 'Q,K,V ← action tokens (causal mask)',
         COLORS['orange_light'], COLORS['orange']),
        ('Block 3: Cross-Attention', 'CA', 'Q ← action,  K,V ← VLM features',
         COLORS['purple_light'], COLORS['purple']),
        ('Block 4: Self-Attention',  'SA', 'Q,K,V ← action tokens (causal mask)',
         COLORS['orange_light'], COLORS['orange']),
    ]
    block_x = 4.0
    block_y0 = 1.6
    block_h = 1.4
    block_w = 7.0
    for i, (title, kind, desc, fc, ec) in enumerate(block_specs):
        y = block_y0 + i * (block_h + 0.18)
        box = FancyBboxPatch((block_x, y), block_w, block_h,
                             boxstyle='round,pad=0.04,rounding_size=0.06',
                             facecolor=fc, alpha=0.5, edgecolor=ec, linewidth=1.5)
        ax.add_patch(box)
        ax.text(block_x + block_w / 2, y + block_h - 0.32,
                title, ha='center', fontsize=11, fontweight='bold', color=ec)
        ax.text(block_x + block_w / 2, y + 0.4,
                desc, ha='center', fontsize=10, color=COLORS['text'])

        # AdaLN(τ) badge on right
        adaln_x = block_x + block_w + 0.2
        adaln = FancyBboxPatch((adaln_x, y + 0.3), 1.2, block_h - 0.6,
                               boxstyle='round,pad=0.02,rounding_size=0.05',
                               facecolor='#fff4e0', edgecolor=COLORS['orange'])
        ax.add_patch(adaln)
        ax.text(adaln_x + 0.6, y + block_h / 2,
                'AdaLN(τ)', ha='center', va='center', fontsize=9, color=COLORS['orange'])

        # CA -> connect to VLM features (curve)
        if kind == 'CA':
            arrow = FancyArrowPatch((2.3, y + block_h / 2),
                                    (block_x + 0.05, y + block_h / 2),
                                    arrowstyle='->', mutation_scale=15,
                                    color=COLORS['purple'], linewidth=1.5,
                                    connectionstyle='arc3,rad=0.0')
            ax.add_patch(arrow)

    # Action token input -> Block 1 (upward)
    arrow_up = FancyArrowPatch((7.5, 1.25), (7.5, 1.55),
                                arrowstyle='->', mutation_scale=15,
                                color=COLORS['gray'], linewidth=1.4)
    ax.add_patch(arrow_up)

    # Output arrow at top
    top_y = block_y0 + len(block_specs) * (block_h + 0.18) - 0.06
    arrow_out = FancyArrowPatch((7.5, top_y), (7.5, top_y + 0.55),
                                arrowstyle='->', mutation_scale=15,
                                color=COLORS['gray'], linewidth=1.4)
    ax.add_patch(arrow_out)

    output_box = FancyBboxPatch((4.0, top_y + 0.6), 7.0, 0.7,
                                boxstyle='round,pad=0.04,rounding_size=0.06',
                                facecolor=COLORS['green_light'], alpha=0.7,
                                edgecolor=COLORS['green'], linewidth=1.6)
    ax.add_patch(output_box)
    ax.text(7.5, top_y + 0.95,
            'Output: velocity v$_θ$  [n=50, action_dim=7]',
            ha='center', va='center', fontsize=11, color=COLORS['text'])

    # τ embedding box (right side)
    tau_box = FancyBboxPatch((12.0, 4.0), 1.7, 1.5,
                             boxstyle='round,pad=0.05,rounding_size=0.08',
                             facecolor='#fff4e0', edgecolor=COLORS['orange'], linewidth=1.5)
    ax.add_patch(tau_box)
    ax.text(12.85, 5.0, 'τ embed', ha='center', fontsize=11, fontweight='bold',
            color=COLORS['orange'])
    ax.text(12.85, 4.4, 'sinusoidal\n+ MLP',
            ha='center', fontsize=9, color=COLORS['text_muted'])

    plt.subplots_adjust(top=0.92, bottom=0.04, left=0.02, right=0.98)
    save_both(fig, 'smolvla_action_expert_blocks')
    plt.close(fig)


# =====================================================
# Figure 2: Pixel shuffle (spatial -> channel)
# =====================================================
def fig_pixel_shuffle():
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    clean_axes(ax)

    fig.suptitle('Pixel Unshuffle: 공간 (spatial) → 채널 (channel) 정보 이동',
                 fontsize=14, fontweight='bold', y=0.97)

    # Left: original 4x4 patches (16 patches, dim d each)
    ax.text(1.5, 7.2, 'Before  [4×4=16 patches, dim d]',
            ha='center', fontsize=11, fontweight='bold', color=COLORS['text'])
    for i in range(4):
        for j in range(4):
            x = 0.1 + j * 0.7
            y = 6.6 - i * 0.7
            # Color groups: 2x2 quadrants share color
            qi = i // 2
            qj = j // 2
            group = qi * 2 + qj
            colors = [COLORS['purple_light'], COLORS['green_light'],
                      COLORS['orange_light'], COLORS['red_light']]
            rect = Rectangle((x, y), 0.65, 0.65,
                             facecolor=colors[group], edgecolor=COLORS['gray'])
            ax.add_patch(rect)

    # Group boundary lines (2x2 blocks)
    for k in [1, 2]:
        if k == 1:
            continue  # 1.4 mark = boundary
        pass
    # Boundary at j=2 and i=2
    # vertical at x = 0.1 + 2*0.7 = 1.5
    ax.plot([1.5, 1.5], [3.8, 7.1], color=COLORS['text'], linewidth=2.0)
    # horizontal at y = 6.6 - 2*0.7 + 0.65 = 5.85 (top of row 2)
    ax.plot([0.0, 2.9], [5.85, 5.85], color=COLORS['text'], linewidth=2.0)

    ax.text(1.45, 3.4, '2×2 인접 patch를\n한 그룹으로',
            ha='center', fontsize=9, color=COLORS['text_muted'], style='italic')

    # Arrow to middle (reshape)
    arrow = FancyArrowPatch((3.5, 5.5), (5.5, 5.5),
                            arrowstyle='-|>', mutation_scale=18,
                            color=COLORS['gray'], linewidth=1.6)
    ax.add_patch(arrow)
    ax.text(4.5, 6.0, 'reshape +\nconcat 채널',
            ha='center', fontsize=10, color=COLORS['text_muted'])

    # Middle: 2x2 super-patches with dim 4d (concat of 4 patches)
    ax.text(7.0, 7.2, 'After reshape  [2×2=4 patches, dim 4d]',
            ha='center', fontsize=11, fontweight='bold', color=COLORS['text'])
    super_colors = [COLORS['purple_light'], COLORS['green_light'],
                    COLORS['orange_light'], COLORS['red_light']]
    super_pos = [(5.7, 6.0), (7.2, 6.0), (5.7, 4.5), (7.2, 4.5)]
    super_labels = ['[p1,p2,\np5,p6]', '[p3,p4,\np7,p8]',
                    '[p9,p10,\np13,p14]', '[p11,p12,\np15,p16]']
    for (x, y), c, lbl in zip(super_pos, super_colors, super_labels):
        rect = Rectangle((x, y), 1.3, 1.3, facecolor=c, edgecolor=COLORS['gray'], linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + 0.65, y + 0.65, lbl, ha='center', va='center',
                fontsize=9, color=COLORS['text'])

    ax.text(7.0, 3.6, '4 patch 정보가\n채널 방향으로 합쳐짐',
            ha='center', fontsize=9, color=COLORS['text_muted'], style='italic')

    # Arrow to right (linear projection)
    arrow2 = FancyArrowPatch((9.0, 5.2), (11.0, 5.2),
                             arrowstyle='-|>', mutation_scale=18,
                             color=COLORS['gray'], linewidth=1.6)
    ax.add_patch(arrow2)
    ax.text(10.0, 5.65, 'Linear\nprojection',
            ha='center', fontsize=10, color=COLORS['text_muted'])

    # Right: compressed back to dim d
    ax.text(13.2, 7.2, 'After projection  [4 patches, dim d]',
            ha='center', fontsize=11, fontweight='bold', color=COLORS['text'])
    out_colors = ['#bfb8e2', '#bfddc9', '#e7c8b7', '#dbb6b6']
    for (x, y), c in zip([(11.4, 6.2), (12.7, 6.2), (11.4, 4.9), (12.7, 4.9)], out_colors):
        rect = Rectangle((x, y), 1.1, 1.0, facecolor=c, edgecolor=COLORS['gray'], linewidth=1.4)
        ax.add_patch(rect)

    # Bottom caption: numerical example
    fig.text(0.5, 0.07,
             '예: 256 patches × dim 768  →  64 patches × dim 768   '
             '(token 수 ¼, 정보 손실 거의 없음)',
             ha='center', fontsize=12, color=COLORS['text'])
    fig.text(0.5, 0.025,
             'Sequence length 제곱에 비례하는 transformer 비용 → 1/16 절감',
             ha='center', fontsize=10, color=COLORS['text_muted'], style='italic')

    plt.subplots_adjust(top=0.92, bottom=0.15)
    save_both(fig, 'smolvla_pixel_shuffle')
    plt.close(fig)


# =====================================================
# Figure 3: Async vs Sync inference timeline
# =====================================================
def fig_async_inference():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5.5),
                                    gridspec_kw={'hspace': 0.5})
    fig.patch.set_facecolor('white')

    fig.suptitle('SmolVLA Async vs Sync Inference Timeline',
                 fontsize=14, fontweight='bold', y=0.99)

    inference_time = 1.4  # arbitrary units
    chunk_duration = 2.0  # action chunk play time

    # ============ Sync mode ============
    ax1.set_title('Sync mode: 학습 중 idle time 발생',
                  fontsize=11, color=COLORS['red'], loc='left')
    cycles = 3
    cur = 0
    for k in range(cycles):
        # Inference (gray)
        ax1.barh(0, inference_time, left=cur, height=0.5,
                 color=COLORS['gray_light'], edgecolor=COLORS['gray'], linewidth=1.0)
        ax1.text(cur + inference_time / 2, 0, 'inference',
                 ha='center', va='center', fontsize=9, color=COLORS['text'])
        cur += inference_time
        # Action chunk play (green)
        ax1.barh(0, chunk_duration, left=cur, height=0.5,
                 color=COLORS['green_light'], edgecolor=COLORS['green'], linewidth=1.0)
        ax1.text(cur + chunk_duration / 2, 0, 'action chunk',
                 ha='center', va='center', fontsize=9, color=COLORS['text'])
        cur += chunk_duration

    ax1.set_xlim(0, cur + 0.5)
    ax1.set_ylim(-0.5, 0.7)
    ax1.set_yticks([])
    ax1.set_xlabel('time →', fontsize=10)
    clean_axes(ax1, keep=('bottom',))
    ax1.set_xticks([])

    # Idle annotation
    ax1.text(inference_time / 2, 0.45, 'robot IDLE',
             ha='center', fontsize=10, color=COLORS['red'],
             fontweight='bold')

    # ============ Async mode ============
    ax2.set_title('Async mode: inference와 action 실행이 병렬',
                  fontsize=11, color=COLORS['green'], loc='left')
    # Two tracks: prediction (top) and execution (bottom)
    cur_pred = 0
    cur_exec = 0
    n_cycles = 4
    pred_offsets = []
    for k in range(n_cycles):
        # Prediction (top track) — async, kicked off when queue threshold hit
        if k == 0:
            pred_start = 0
        else:
            pred_start = pred_offsets[-1] + chunk_duration * 0.7
        ax2.barh(0.6, inference_time, left=pred_start, height=0.4,
                 color=COLORS['gray_light'], edgecolor=COLORS['gray'], linewidth=1.0)
        ax2.text(pred_start + inference_time / 2, 0.6, f'inf {k+1}',
                 ha='center', va='center', fontsize=9, color=COLORS['text'])
        pred_offsets.append(pred_start)

    # Execution (bottom track) — continuous
    for k in range(n_cycles):
        exec_start = k * chunk_duration
        ax2.barh(-0.2, chunk_duration, left=exec_start, height=0.4,
                 color=COLORS['green_light'], edgecolor=COLORS['green'], linewidth=1.0)
        ax2.text(exec_start + chunk_duration / 2, -0.2, f'chunk {k+1}',
                 ha='center', va='center', fontsize=9, color=COLORS['text'])

    # Track labels
    ax2.text(-0.3, 0.6, 'Policy\nserver', ha='right', va='center',
             fontsize=9, color=COLORS['text_muted'])
    ax2.text(-0.3, -0.2, 'Robot\nclient', ha='right', va='center',
             fontsize=9, color=COLORS['text_muted'])

    ax2.set_xlim(-1.0, n_cycles * chunk_duration + 0.5)
    ax2.set_ylim(-0.7, 1.1)
    ax2.set_yticks([])
    ax2.set_xlabel('time →', fontsize=10)
    clean_axes(ax2, keep=('bottom',))
    ax2.set_xticks([])

    # Result annotation
    ax2.text(n_cycles * chunk_duration / 2, 1.0,
             'robot은 항상 실행 중. inference는 다음 chunk를 미리 준비',
             ha='center', fontsize=10, color=COLORS['green'], fontweight='bold')

    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.10, right=0.95)
    save_both(fig, 'smolvla_async_inference')
    plt.close(fig)


# =====================================================
# Figure 4: VLM layer skipping
# =====================================================
def fig_layer_skipping():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.set_xticks([])
    ax.set_yticks([])
    clean_axes(ax)

    fig.suptitle('SmolVLA: SmolLM-2 layer skipping (앞 절반만 사용)',
                 fontsize=14, fontweight='bold', y=0.97)

    # 16 layers as stacked boxes
    total_layers = 16
    use_layers = 8
    box_w = 8.0
    box_h = 0.35
    x0 = 3.0
    y0 = 0.5

    for k in range(total_layers):
        y = y0 + k * (box_h + 0.05)
        if k < use_layers:
            fc = COLORS['purple_light']
            ec = COLORS['purple']
            alpha = 0.7
        else:
            fc = COLORS['gray_light']
            ec = COLORS['gray']
            alpha = 0.35

        rect = Rectangle((x0, y), box_w, box_h,
                         facecolor=fc, edgecolor=ec, linewidth=1.0, alpha=alpha)
        ax.add_patch(rect)
        ax.text(x0 + box_w / 2, y + box_h / 2,
                f'Layer {k + 1}', ha='center', va='center',
                fontsize=9, color=COLORS['text'] if k < use_layers else COLORS['text_muted'])

    # Right side: feature extraction point
    extract_y = y0 + (use_layers - 1) * (box_h + 0.05) + box_h
    arrow = FancyArrowPatch((x0 + box_w + 0.1, extract_y),
                            (x0 + box_w + 1.5, extract_y),
                            arrowstyle='-|>', mutation_scale=16,
                            color=COLORS['orange'], linewidth=2.0)
    ax.add_patch(arrow)
    ax.text(x0 + box_w + 0.8, extract_y + 0.3,
            'features at\nlayer N=L/2=8',
            ha='center', va='bottom', fontsize=10, color=COLORS['orange'],
            fontweight='bold')

    out_box = FancyBboxPatch((x0 + box_w + 1.7, extract_y - 0.4), 2.5, 0.8,
                             boxstyle='round,pad=0.04,rounding_size=0.06',
                             facecolor=COLORS['green_light'], alpha=0.7,
                             edgecolor=COLORS['green'], linewidth=1.5)
    ax.add_patch(out_box)
    ax.text(x0 + box_w + 1.7 + 1.25, extract_y, 'VLM features\n(perception)',
            ha='center', va='center', fontsize=10, color=COLORS['text'])

    # Left side label: VLM input
    in_box = FancyBboxPatch((0.3, 0.5), 2.2, 8.4,
                            boxstyle='round,pad=0.04,rounding_size=0.06',
                            facecolor='#fafafa', edgecolor=COLORS['gray'], linewidth=1.2)
    ax.add_patch(in_box)
    ax.text(1.4, 7.5, 'Input', fontsize=12, fontweight='bold', ha='center',
            color=COLORS['text'])
    ax.text(1.4, 6.6, 'images\n(64 tok/cam)\n+ text\n+ state', ha='center',
            fontsize=10, color=COLORS['text'])

    # Arrow from input into layer 1
    arrow_in = FancyArrowPatch((2.55, y0 + box_h / 2), (x0 - 0.05, y0 + box_h / 2),
                                arrowstyle='-|>', mutation_scale=14,
                                color=COLORS['gray'], linewidth=1.3)
    ax.add_patch(arrow_in)

    # Annotations: used / discarded brackets
    used_top = y0 + use_layers * (box_h + 0.05) - 0.05
    ax.annotate('', xy=(x0 - 0.4, y0), xytext=(x0 - 0.4, used_top),
                arrowprops=dict(arrowstyle='-', color=COLORS['purple'], linewidth=2.5))
    ax.text(x0 - 0.6, (y0 + used_top) / 2, '사용\n(N = 8)',
            ha='right', va='center', fontsize=10, color=COLORS['purple'],
            fontweight='bold')

    discard_top = y0 + total_layers * (box_h + 0.05)
    discard_bot = used_top
    ax.annotate('', xy=(x0 - 0.4, discard_bot), xytext=(x0 - 0.4, discard_top),
                arrowprops=dict(arrowstyle='-', color=COLORS['gray'], linewidth=2.5))
    ax.text(x0 - 0.6, (discard_top + discard_bot) / 2, '버림\n(8 layers)',
            ha='right', va='center', fontsize=10, color=COLORS['gray'])

    # Bottom caption
    fig.text(0.5, 0.04,
             '마지막 N층은 자연어 generation에 특화 → robot 제어엔 불필요.  '
             '계산 ½, 성능 유지.',
             ha='center', fontsize=11, color=COLORS['text'])

    plt.subplots_adjust(top=0.92, bottom=0.10)
    save_both(fig, 'smolvla_layer_skipping')
    plt.close(fig)


if __name__ == '__main__':
    import matplotlib
    fig_action_expert_blocks()
    fig_pixel_shuffle()
    fig_async_inference()
    fig_layer_skipping()
    print('All SmolVLA figures generated.')
