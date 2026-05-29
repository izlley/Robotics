"""Generate all SVG figures for pi0.md."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _style import setup, COLORS, save_both, clean_axes, add_box

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
from scipy.stats import beta as beta_dist

setup()

# =====================================================
# Figure 1: tau distribution (shifted Beta)
# =====================================================
def fig_tau_distribution():
    s = 0.999
    alpha, beta_p = 1.5, 1.0
    tau = np.linspace(0, s, 500)
    u = (s - tau) / s
    p = beta_dist.pdf(u, alpha, beta_p) / s

    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor('white')

    ax.fill_between(tau, 0, p, color=COLORS['purple_light'], alpha=0.55)
    ax.plot(tau, p, color=COLORS['purple'], linewidth=2.0)

    ax.plot(0, 1.5, marker='o', markersize=10, markerfacecolor='white',
            markeredgecolor=COLORS['orange'], markeredgewidth=2.0, zorder=10)
    ax.annotate('peak (τ=0)\n1.5',
                xy=(0.02, 1.45), xytext=(0.07, 1.30),
                fontsize=12, color=COLORS['text'], ha='left', va='top')

    ax.axvline(x=s, ymin=0, ymax=0.93, color=COLORS['red'],
               linestyle='--', linewidth=1.8, alpha=0.8)
    ax.annotate('τ = 0.999\n절단',
                xy=(s - 0.005, 1.4), xytext=(s - 0.01, 1.45),
                fontsize=11, color=COLORS['red'], ha='right', va='top')

    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0, 1.75)
    ax.set_xlabel('τ', fontsize=14)
    ax.set_ylabel('p(τ)', fontsize=14, rotation=0, labelpad=18, va='center')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 1.5])
    clean_axes(ax, keep=('bottom', 'left'))
    ax.tick_params(labelsize=11)

    fig.suptitle('π$_0$의 최종 τ 분포', fontsize=15, fontweight='bold', y=0.97)
    ax.set_title(r'p(τ) = Beta( (0.999 - τ)/0.999 ; 1.5, 1)',
                 fontsize=12, color=COLORS['text'], pad=14)

    quarters = [(0.0, 0.25, '≈ 40%'),
                (0.25, 0.5, '≈ 30%'),
                (0.5, 0.75, '≈ 20%'),
                (0.75, 1.0, '≈ 10%')]
    box_y = -0.32
    for x0, x1, label in quarters:
        rect = Rectangle((x0, box_y), x1 - x0, 0.13,
                         facecolor=COLORS['box_fill'], edgecolor=COLORS['box_edge'],
                         transform=ax.get_xaxis_transform(), clip_on=False)
        ax.add_patch(rect)
        ax.text((x0 + x1) / 2, box_y + 0.065, label,
                transform=ax.get_xaxis_transform(),
                ha='center', va='center', fontsize=11)

    ax.text(0.5, -0.50,
            'τ ∈ [0, 0.25] 구간이 가장 자주 샘플링됨 (어려운 영역에 집중)',
            transform=ax.get_xaxis_transform(),
            ha='center', va='center', fontsize=11, style='italic', color=COLORS['text_muted'])

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_both(fig, 'pi0_tau_distribution')
    plt.close(fig)


# =====================================================
# Figure 2: linear path noise -> action
# =====================================================
def fig_linear_path():
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor('white')

    eps = np.array([1.5, 2.2])
    A_t = np.array([9.5, 6.5])
    taus = [0.0, 0.25, 0.5, 0.75, 1.0]
    pts = [eps + tau * (A_t - eps) for tau in taus]

    ax.plot([eps[0], A_t[0]], [eps[1], A_t[1]],
            color=COLORS['purple'], linestyle='--', linewidth=1.6, alpha=0.6, zorder=1)

    for i in range(len(pts) - 1):
        p1, p2 = pts[i], pts[i + 1]
        d = p2 - p1
        u = d / np.linalg.norm(d)
        start = p1 + u * 0.18
        end = p2 - u * 0.18
        arrow = FancyArrowPatch(start, end,
                                arrowstyle='-|>', mutation_scale=18,
                                color=COLORS['green'], linewidth=2.0, zorder=2)
        ax.add_patch(arrow)

    ax.scatter(*eps, s=380, c=COLORS['gray_light'], edgecolors='#8a8884', linewidths=1.5, zorder=3)
    ax.scatter(*A_t, s=380, c=COLORS['green_light'], edgecolors=COLORS['green'], linewidths=1.5, zorder=3)
    for pt in pts[1:-1]:
        ax.scatter(*pt, s=110, c=COLORS['purple'], zorder=4)

    ax.annotate('ε (τ=0)', xy=eps, xytext=(eps[0] - 0.05, eps[1] - 0.65),
                fontsize=13, color=COLORS['text_muted'], ha='center')
    ax.annotate('A$_t$ (τ=1)', xy=A_t, xytext=(A_t[0] + 0.15, A_t[1] - 0.55),
                fontsize=13, color=COLORS['green'], ha='left')

    for label, pt in zip(['τ=0.25', 'τ=0.5', 'τ=0.75'], pts[1:-1]):
        ax.annotate(label, xy=pt, xytext=(pt[0] - 0.7, pt[1] + 0.45),
                    fontsize=11, color=COLORS['text_muted'], ha='left')

    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    clean_axes(ax, keep=('bottom', 'left'))
    ax.spines['bottom'].set_color(COLORS['text'])
    ax.spines['left'].set_color(COLORS['text'])
    ax.text(11.2, 0.15, 'x', fontsize=13, color=COLORS['text'])
    ax.text(0.15, 8.7, 'y', fontsize=13, color=COLORS['text'])

    fig.suptitle('2차원 공간에서 본 직선 경로', fontsize=15, fontweight='bold', y=0.96)
    ax.set_title('노이즈 ε 에서 action A$_t$ 로 가는 직선',
                 fontsize=12, color=COLORS['text_muted'], pad=10)

    fig.text(0.5, 0.10,
             '모든 지점에서 velocity 벡터(초록)가 방향과 크기 모두 동일 = $A_t - ε$',
             ha='center', fontsize=12, color=COLORS['text'])
    fig.text(0.5, 0.025,
             '직선이라서 미분(접선 방향)이 일정한 상수 벡터',
             ha='center', fontsize=11, color=COLORS['text_muted'], style='italic')

    plt.subplots_adjust(top=0.86, bottom=0.20)
    save_both(fig, 'pi0_linear_path')
    plt.close(fig)


# =====================================================
# Figure 3: MoE single transformer (two weight sets)
# =====================================================
def fig_moe_architecture():
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    clean_axes(ax)

    fig.suptitle('π$_0$ MoE: 단일 transformer 안의 두 weight set',
                 fontsize=15, fontweight='bold', y=0.96)

    # Token row at top
    tokens = [
        (0.5, 6.5, 1.6, 0.7, 'image\npatches', COLORS['purple_light']),
        (2.3, 6.5, 1.6, 0.7, 'language\ntokens', COLORS['purple_light']),
        (4.1, 6.5, 1.4, 0.7, 'state q$_t$\n(1 token)', COLORS['green_light']),
        (5.7, 6.5, 1.7, 0.7, 'noisy A$^τ$\n(50 tokens)', COLORS['green_light']),
    ]
    for x, y, w, h, label, color in tokens:
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle='round,pad=0.02,rounding_size=0.06',
                             facecolor=color, edgecolor=COLORS['gray'], linewidth=1.2)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=10)

    ax.text(0.5, 7.5, 'Input tokens:', fontsize=11, fontweight='bold', color=COLORS['text'])

    # Routing arrows
    for x_token, x_expert in [(1.3, 2.5), (3.1, 2.5), (4.8, 8.5), (6.55, 8.5)]:
        arrow = FancyArrowPatch((x_token, 6.45), (x_expert, 5.05),
                                arrowstyle='->', mutation_scale=14,
                                color=COLORS['gray'], linewidth=1.1, alpha=0.75)
        ax.add_patch(arrow)

    # Two expert blocks
    vlm_box = FancyBboxPatch((0.5, 2.5), 4.5, 2.5,
                             boxstyle='round,pad=0.05,rounding_size=0.1',
                             facecolor=COLORS['purple_light'], alpha=0.4,
                             edgecolor=COLORS['purple'], linewidth=2.0)
    ax.add_patch(vlm_box)
    ax.text(2.75, 4.6, 'VLM Expert', ha='center', fontsize=13, fontweight='bold',
            color=COLORS['purple'])
    ax.text(2.75, 4.1, '(PaliGemma 3B, Gemma 2B based)',
            ha='center', fontsize=10, color=COLORS['text'], style='italic')
    ax.text(2.75, 3.6, 'width = 2048\nmlp = 16,384', ha='center', fontsize=10,
            color=COLORS['text'])
    ax.text(2.75, 2.85, 'W$_Q$, W$_K$, W$_V$, FFN 모두 자기 weight',
            ha='center', fontsize=9, color=COLORS['text_muted'])

    action_box = FancyBboxPatch((6.5, 2.5), 4.5, 2.5,
                                boxstyle='round,pad=0.05,rounding_size=0.1',
                                facecolor=COLORS['green_light'], alpha=0.5,
                                edgecolor=COLORS['green'], linewidth=2.0)
    ax.add_patch(action_box)
    ax.text(8.75, 4.6, 'Action Expert', ha='center', fontsize=13, fontweight='bold',
            color=COLORS['green'])
    ax.text(8.75, 4.1, '(300M, scratch init)',
            ha='center', fontsize=10, color=COLORS['text'], style='italic')
    ax.text(8.75, 3.6, 'width = 1024 (절반)\nmlp = 4,096 (¼)',
            ha='center', fontsize=10, color=COLORS['text'])
    ax.text(8.75, 2.85, '추론 시 10번 forward → 작을수록 빠름',
            ha='center', fontsize=9, color=COLORS['text_muted'])

    # Shared self-attention layer
    sa_box = FancyBboxPatch((1.0, 0.7), 9.5, 1.2,
                            boxstyle='round,pad=0.05,rounding_size=0.1',
                            facecolor='#fff4e0', edgecolor=COLORS['orange'], linewidth=1.8)
    ax.add_patch(sa_box)
    ax.text(5.75, 1.35,
            'Self-Attention layer: 두 expert가 만나는 유일한 곳',
            ha='center', fontsize=12, fontweight='bold', color=COLORS['orange'])
    ax.text(5.75, 0.95, 'softmax(Q · K$^T$) · V — Q/K/V는 각 expert weight로 계산',
            ha='center', fontsize=10, color=COLORS['text'])

    # Arrows from experts down to SA
    for x_start in [2.75, 8.75]:
        arrow = FancyArrowPatch((x_start, 2.4), (x_start, 1.95),
                                arrowstyle='->', mutation_scale=18,
                                color=COLORS['gray'], linewidth=1.5)
        ax.add_patch(arrow)

    plt.subplots_adjust(top=0.92, bottom=0.04, left=0.02, right=0.98)
    save_both(fig, 'pi0_moe_architecture')
    plt.close(fig)


# =====================================================
# Figure 4: Block-wise causal attention mask
# =====================================================
def fig_attention_mask():
    fig, ax = plt.subplots(figsize=(8.5, 7))
    fig.patch.set_facecolor('white')

    # Block sizes (n=64 for visual, but conceptually variable)
    blocks = [
        ('Block 1\n(img+lang)', 20),
        ('Block 2\n(state)',     4),
        ('Block 3\n(action chunk)', 30),
    ]
    block_starts = [0]
    for _, sz in blocks[:-1]:
        block_starts.append(block_starts[-1] + sz)
    total = block_starts[-1] + blocks[-1][1]

    # Build mask matrix: 1 = can attend, 0 = blocked
    mask = np.zeros((total, total))
    block_idx_for_pos = []
    for bi, (_, sz) in enumerate(blocks):
        for _ in range(sz):
            block_idx_for_pos.append(bi)

    for i in range(total):
        for j in range(total):
            bi, bj = block_idx_for_pos[i], block_idx_for_pos[j]
            if bj <= bi:  # can see previous + own block
                mask[i, j] = 1.0
            else:
                mask[i, j] = 0.0

    # Plot
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'attn', [(0.0, '#fdfdfd'), (1.0, COLORS['purple_light'])])
    ax.imshow(mask, cmap=cmap, origin='upper', aspect='equal', alpha=0.85)

    # Add block boundary lines
    for s in block_starts[1:]:
        ax.axhline(s - 0.5, color=COLORS['text'], linewidth=1.2)
        ax.axvline(s - 0.5, color=COLORS['text'], linewidth=1.2)
    ax.axhline(total - 0.5, color=COLORS['text'], linewidth=1.2)
    ax.axvline(total - 0.5, color=COLORS['text'], linewidth=1.2)
    ax.axhline(-0.5, color=COLORS['text'], linewidth=1.2)
    ax.axvline(-0.5, color=COLORS['text'], linewidth=1.2)

    # Block labels (centers)
    centers = []
    pos = 0
    for label, sz in blocks:
        centers.append(pos + sz / 2 - 0.5)
        pos += sz
    ax.set_xticks(centers)
    ax.set_xticklabels([b[0].replace('\n', ' ') for b in blocks], fontsize=10)
    ax.set_yticks(centers)
    ax.set_yticklabels([b[0].replace('\n', ' ') for b in blocks], fontsize=10, rotation=90, va='center')

    # Within-block label annotations
    for bi, (label, sz) in enumerate(blocks):
        cx = block_starts[bi] + sz / 2 - 0.5
        ax.text(cx, cx, 'bi-di', ha='center', va='center',
                fontsize=11, color=COLORS['purple'], fontweight='bold')

    # Bidirectional & blocked annotations between blocks
    for i in range(len(blocks)):
        for j in range(len(blocks)):
            if i == j:
                continue
            cx = block_starts[j] + blocks[j][1] / 2 - 0.5
            cy = block_starts[i] + blocks[i][1] / 2 - 0.5
            if j < i:
                ax.text(cx, cy, 'OK', ha='center', va='center',
                        fontsize=13, color=COLORS['green'], fontweight='bold')
            else:
                ax.text(cx, cy, 'NO', ha='center', va='center',
                        fontsize=13, color=COLORS['red'], fontweight='bold')

    fig.suptitle('π$_0$ Block-wise Causal Attention Mask',
                 fontsize=14, fontweight='bold', y=0.99)
    ax.set_title('Within block: bidirectional. Between blocks: causal (previous only)',
                 fontsize=11, color=COLORS['text_muted'], pad=10)
    ax.set_xlabel('Key / Value (attend to)', fontsize=11)
    ax.set_ylabel('Query (attending from)', fontsize=11)

    # Legend
    legend_items = [
        ('bi-di', COLORS['purple'], 'Within block (bidirectional)'),
        ('OK',    COLORS['green'],  'Can attend (previous block)'),
        ('NO',    COLORS['red'],    'Blocked (future block)'),
    ]
    for k, (sym, color, desc) in enumerate(legend_items):
        fig.text(0.07, 0.10 - k * 0.025, f'  {sym}  ',
                 fontsize=12, color=color, va='center', fontweight='bold')
        fig.text(0.18, 0.10 - k * 0.025, desc,
                 fontsize=10, color=COLORS['text'], va='center')

    plt.subplots_adjust(top=0.91, bottom=0.20, left=0.18, right=0.95)
    save_both(fig, 'pi0_attention_mask')
    plt.close(fig)


if __name__ == '__main__':
    import matplotlib
    fig_tau_distribution()
    fig_linear_path()
    fig_moe_architecture()
    fig_attention_mask()
    print('All pi0 figures generated.')
