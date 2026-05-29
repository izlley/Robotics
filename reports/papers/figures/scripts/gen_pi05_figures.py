"""Generate all SVG figures for pi0_5.md."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _style import setup, COLORS, save_both, clean_axes

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch, Wedge

setup()


# =====================================================
# Figure 1: Hierarchical inference (subtask -> action)
# =====================================================
def fig_hierarchical_inference():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    clean_axes(ax)

    fig.suptitle('π0.5 Hierarchical Inference: 같은 모델, 두 단계 추론',
                 fontsize=14, fontweight='bold', y=0.97)

    # Input
    in_box = FancyBboxPatch((0.3, 3.0), 2.5, 2.0,
                            boxstyle='round,pad=0.06,rounding_size=0.1',
                            facecolor='#fafafa', edgecolor=COLORS['gray'], linewidth=1.3)
    ax.add_patch(in_box)
    ax.text(1.55, 4.55, 'Input', fontsize=12, fontweight='bold', ha='center',
            color=COLORS['text'])
    ax.text(1.55, 3.95, 'observation o$_t$\n+ task ℓ', ha='center', fontsize=10,
            color=COLORS['text'])
    ax.text(1.55, 3.25, '"clean the kitchen"',
            ha='center', fontsize=9, color=COLORS['text_muted'], style='italic')

    # Stage 1 box (high-level)
    s1_box = FancyBboxPatch((3.6, 4.5), 4.5, 2.6,
                            boxstyle='round,pad=0.06,rounding_size=0.1',
                            facecolor=COLORS['purple_light'], alpha=0.5,
                            edgecolor=COLORS['purple'], linewidth=1.8)
    ax.add_patch(s1_box)
    ax.text(5.85, 6.7, 'Stage 1: HIGH-LEVEL',
            ha='center', fontsize=11, fontweight='bold', color=COLORS['purple'])
    ax.text(5.85, 6.2, 'autoregressive text decode',
            ha='center', fontsize=10, color=COLORS['text'], style='italic')
    ax.text(5.85, 5.65, 'π$_θ$(ℓ̂ | o$_t$, ℓ)',
            ha='center', fontsize=11, color=COLORS['text'])
    ax.text(5.85, 5.05, '주기 ~ 2 sec',
            ha='center', fontsize=9, color=COLORS['text_muted'])

    # Subtask label (intermediate)
    sub_box = FancyBboxPatch((8.7, 4.7), 2.8, 2.2,
                             boxstyle='round,pad=0.05,rounding_size=0.08',
                             facecolor='#fff4e0', edgecolor=COLORS['orange'], linewidth=1.5)
    ax.add_patch(sub_box)
    ax.text(10.1, 6.5, 'subtask ℓ̂', ha='center', fontsize=11, fontweight='bold',
            color=COLORS['orange'])
    ax.text(10.1, 5.95, '"pick up the\nplate"',
            ha='center', fontsize=10, color=COLORS['text'], style='italic')
    ax.text(10.1, 5.0, '(intermediate\nreasoning)',
            ha='center', fontsize=8, color=COLORS['text_muted'])

    # Stage 2 box (low-level)
    s2_box = FancyBboxPatch((3.6, 0.8), 4.5, 2.6,
                            boxstyle='round,pad=0.06,rounding_size=0.1',
                            facecolor=COLORS['green_light'], alpha=0.6,
                            edgecolor=COLORS['green'], linewidth=1.8)
    ax.add_patch(s2_box)
    ax.text(5.85, 3.0, 'Stage 2: LOW-LEVEL',
            ha='center', fontsize=11, fontweight='bold', color=COLORS['green'])
    ax.text(5.85, 2.5, 'flow matching ODE 10 step',
            ha='center', fontsize=10, color=COLORS['text'], style='italic')
    ax.text(5.85, 1.95, 'π$_θ$(A | o$_t$, ℓ̂)',
            ha='center', fontsize=11, color=COLORS['text'])
    ax.text(5.85, 1.35, '주기 ~ 50 Hz',
            ha='center', fontsize=9, color=COLORS['text_muted'])

    # Output
    out_box = FancyBboxPatch((9.2, 0.95), 4.3, 2.0,
                             boxstyle='round,pad=0.05,rounding_size=0.08',
                             facecolor=COLORS['gray_light'], alpha=0.7,
                             edgecolor=COLORS['gray'], linewidth=1.4)
    ax.add_patch(out_box)
    ax.text(11.35, 2.6, 'Output', ha='center', fontsize=11, fontweight='bold',
            color=COLORS['text'])
    ax.text(11.35, 2.0, 'action chunk A',
            ha='center', fontsize=10, color=COLORS['text'])
    ax.text(11.35, 1.4, '[50 steps × 18-dim]',
            ha='center', fontsize=9, color=COLORS['text_muted'])

    # Arrows
    # Input -> Stage 1
    a1 = FancyArrowPatch((2.85, 4.5), (3.55, 5.5),
                         arrowstyle='-|>', mutation_scale=18,
                         color=COLORS['gray'], linewidth=1.5)
    ax.add_patch(a1)
    # Stage 1 -> subtask
    a2 = FancyArrowPatch((8.15, 5.85), (8.65, 5.85),
                         arrowstyle='-|>', mutation_scale=18,
                         color=COLORS['orange'], linewidth=1.8)
    ax.add_patch(a2)
    # Subtask -> Stage 2 (down)
    a3 = FancyArrowPatch((10.1, 4.65), (6.5, 3.4),
                         arrowstyle='-|>', mutation_scale=18,
                         color=COLORS['orange'], linewidth=1.5,
                         connectionstyle='arc3,rad=-0.2')
    ax.add_patch(a3)
    # Input -> Stage 2 (also feeds o_t)
    a4 = FancyArrowPatch((2.85, 3.5), (3.55, 2.1),
                         arrowstyle='-|>', mutation_scale=15,
                         color=COLORS['gray'], linewidth=1.3,
                         connectionstyle='arc3,rad=-0.15')
    ax.add_patch(a4)
    # Stage 2 -> output
    a5 = FancyArrowPatch((8.15, 2.1), (9.15, 1.95),
                         arrowstyle='-|>', mutation_scale=18,
                         color=COLORS['green'], linewidth=1.8)
    ax.add_patch(a5)

    # Note: same model
    note = FancyBboxPatch((4.8, 7.4), 4.0, 0.5,
                          boxstyle='round,pad=0.02,rounding_size=0.04',
                          facecolor='#fff8e7', edgecolor='#d4a05a', linewidth=1.0)
    ax.add_patch(note)
    ax.text(6.8, 7.65,
            'Stage 1과 Stage 2는 같은 모델 π$_θ$ (다른 token type)',
            ha='center', va='center', fontsize=10, color='#8b6914',
            fontweight='bold')

    plt.subplots_adjust(top=0.92, bottom=0.03, left=0.02, right=0.98)
    save_both(fig, 'pi05_hierarchical_inference')
    plt.close(fig)


# =====================================================
# Figure 2: Hybrid FAST + Flow training & inference
# =====================================================
def fig_hybrid_fast_flow():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5))
    fig.patch.set_facecolor('white')
    fig.suptitle('π0.5 Hybrid: FAST discrete pre-train + Flow matching post-train',
                 fontsize=14, fontweight='bold', y=0.98)

    # ============ Pre-training (top axes) ============
    ax1.set_title('Pre-training (280k steps): FAST tokens only,  α = 0',
                  fontsize=11, color=COLORS['purple'], loc='left', pad=8)
    ax1.set_xlim(0, 14)
    ax1.set_ylim(0, 4)
    ax1.set_xticks([])
    ax1.set_yticks([])
    clean_axes(ax1)

    # Data input
    in_box = FancyBboxPatch((0.3, 1.2), 2.4, 1.6,
                            boxstyle='round,pad=0.05,rounding_size=0.08',
                            facecolor='#fafafa', edgecolor=COLORS['gray'])
    ax1.add_patch(in_box)
    ax1.text(1.5, 2.4, 'Multi-source\ndata',
             ha='center', va='center', fontsize=10, color=COLORS['text'])
    ax1.text(1.5, 1.55, 'MM+ME+CE\n+HL+WD',
             ha='center', va='center', fontsize=9, color=COLORS['text_muted'])

    # FAST tokenize
    ft_box = FancyBboxPatch((3.4, 1.2), 2.4, 1.6,
                            boxstyle='round,pad=0.05,rounding_size=0.08',
                            facecolor=COLORS['orange_light'], alpha=0.6,
                            edgecolor=COLORS['orange'])
    ax1.add_patch(ft_box)
    ax1.text(4.6, 2.4, 'FAST\ntokenizer',
             ha='center', va='center', fontsize=10, fontweight='bold',
             color=COLORS['orange'])
    ax1.text(4.6, 1.55, 'action → discrete\ntext token',
             ha='center', va='center', fontsize=9, color=COLORS['text_muted'])

    # Transformer
    tr_box = FancyBboxPatch((6.5, 1.2), 3.0, 1.6,
                            boxstyle='round,pad=0.05,rounding_size=0.08',
                            facecolor=COLORS['purple_light'], alpha=0.5,
                            edgecolor=COLORS['purple'])
    ax1.add_patch(tr_box)
    ax1.text(8.0, 2.4, 'Transformer\n(VLM expert)',
             ha='center', va='center', fontsize=10, fontweight='bold',
             color=COLORS['purple'])
    ax1.text(8.0, 1.55, 'next-token\nprediction',
             ha='center', va='center', fontsize=9, color=COLORS['text_muted'])

    # Loss
    loss_box = FancyBboxPatch((10.5, 1.2), 3.0, 1.6,
                              boxstyle='round,pad=0.05,rounding_size=0.08',
                              facecolor=COLORS['red_light'], alpha=0.5,
                              edgecolor=COLORS['red'])
    ax1.add_patch(loss_box)
    ax1.text(12.0, 2.4, 'Cross-Entropy',
             ha='center', va='center', fontsize=10, fontweight='bold',
             color=COLORS['red'])
    ax1.text(12.0, 1.55, 'L = -log p(y | x)',
             ha='center', va='center', fontsize=10, color=COLORS['text'],
             family='monospace')

    # Arrows
    for x0, x1 in [(2.7, 3.35), (5.85, 6.45), (9.55, 10.45)]:
        ax1.add_patch(FancyArrowPatch((x0, 2.0), (x1, 2.0),
                                      arrowstyle='-|>', mutation_scale=15,
                                      color=COLORS['gray'], linewidth=1.4))

    ax1.text(7.0, 0.4,
             '결과: language semantics 학습 보존 (LLM-style)  |  Action expert 비활성',
             ha='center', fontsize=10, color=COLORS['text_muted'], style='italic')

    # ============ Post-training (bottom axes) ============
    ax2.set_title('Post-training (80k steps): FAST + Flow Expert,  α = 10.0',
                  fontsize=11, color=COLORS['green'], loc='left', pad=8)
    ax2.set_xlim(0, 14)
    ax2.set_ylim(0, 4)
    ax2.set_xticks([])
    ax2.set_yticks([])
    clean_axes(ax2)

    # Input
    in_box2 = FancyBboxPatch((0.3, 1.2), 2.4, 1.6,
                             boxstyle='round,pad=0.05,rounding_size=0.08',
                             facecolor='#fafafa', edgecolor=COLORS['gray'])
    ax2.add_patch(in_box2)
    ax2.text(1.5, 2.4, 'MM-focused\ndata + VI',
             ha='center', va='center', fontsize=10, color=COLORS['text'])
    ax2.text(1.5, 1.55, '(human verbal\ninstructions)',
             ha='center', va='center', fontsize=9, color=COLORS['text_muted'])

    # Two parallel paths
    # Top path: FAST text token (cross-entropy)
    fast_box = FancyBboxPatch((3.4, 2.7), 2.8, 1.1,
                              boxstyle='round,pad=0.04,rounding_size=0.06',
                              facecolor=COLORS['orange_light'], alpha=0.5,
                              edgecolor=COLORS['orange'])
    ax2.add_patch(fast_box)
    ax2.text(4.8, 3.25, 'FAST tokens\n(text path)',
             ha='center', va='center', fontsize=10, color=COLORS['orange'])

    ce_box = FancyBboxPatch((7.0, 2.7), 2.5, 1.1,
                            boxstyle='round,pad=0.04,rounding_size=0.06',
                            facecolor=COLORS['red_light'], alpha=0.5,
                            edgecolor=COLORS['red'])
    ax2.add_patch(ce_box)
    ax2.text(8.25, 3.25, 'Cross-Entropy',
             ha='center', va='center', fontsize=10, color=COLORS['red'])

    # Bottom path: Flow expert
    fl_box = FancyBboxPatch((3.4, 0.2), 2.8, 1.1,
                            boxstyle='round,pad=0.04,rounding_size=0.06',
                            facecolor=COLORS['green_light'], alpha=0.5,
                            edgecolor=COLORS['green'])
    ax2.add_patch(fl_box)
    ax2.text(4.8, 0.75, 'Flow expert\n(continuous)',
             ha='center', va='center', fontsize=10, color=COLORS['green'])

    flm_box = FancyBboxPatch((7.0, 0.2), 2.5, 1.1,
                             boxstyle='round,pad=0.04,rounding_size=0.06',
                             facecolor=COLORS['red_light'], alpha=0.5,
                             edgecolor=COLORS['red'])
    ax2.add_patch(flm_box)
    ax2.text(8.25, 0.75, 'Flow MSE\n× α=10',
             ha='center', va='center', fontsize=10, color=COLORS['red'])

    # Sum
    sum_box = FancyBboxPatch((10.4, 1.2), 3.2, 1.6,
                             boxstyle='round,pad=0.05,rounding_size=0.08',
                             facecolor='#fff4e0', edgecolor='#d4a05a', linewidth=1.4)
    ax2.add_patch(sum_box)
    ax2.text(12.0, 2.35, 'Joint loss',
             ha='center', fontsize=11, fontweight='bold', color='#8b6914')
    ax2.text(12.0, 1.7, 'L = L$_{CE}$ + α · L$_{flow}$',
             ha='center', fontsize=10, color=COLORS['text'])

    # Arrows
    for x0, x1, y0, y1 in [(2.7, 3.35, 2.0, 3.25),
                           (2.7, 3.35, 2.0, 0.75),
                           (6.25, 6.95, 3.25, 3.25),
                           (6.25, 6.95, 0.75, 0.75),
                           (9.55, 10.35, 3.25, 2.3),
                           (9.55, 10.35, 0.75, 1.7)]:
        ax2.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                      arrowstyle='-|>', mutation_scale=14,
                                      color=COLORS['gray'], linewidth=1.2))

    plt.subplots_adjust(top=0.92, bottom=0.04, left=0.04, right=0.98, hspace=0.4)
    save_both(fig, 'pi05_hybrid_fast_flow')
    plt.close(fig)


# =====================================================
# Figure 3: Data source mixture (pi chart-like + bar)
# =====================================================
def fig_data_mixture():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                    gridspec_kw={'width_ratios': [1.0, 1.2]})
    fig.patch.set_facecolor('white')
    fig.suptitle('π0.5 Pre-training Data: 5+ source heterogeneous co-training',
                 fontsize=14, fontweight='bold', y=0.99)

    # Pie chart of 5 sources
    labels = ['MM (Mobile Manipulator)\n2.4%', 'ME (Multi-Env)\n~35%',
              'CE (Cross-Embodiment)\n~30%', 'HL (High-Level)\n~15%',
              'WD (Web Data)\n~17.6%']
    sizes = [2.4, 35.0, 30.0, 15.0, 17.6]
    pie_colors = [COLORS['orange'], COLORS['purple_light'],
                  COLORS['green_light'], '#f5d4a0', COLORS['red_light']]
    explode = [0.12, 0, 0, 0, 0]

    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=pie_colors, explode=explode,
        autopct='%.1f%%', startangle=110,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        textprops={'fontsize': 9})
    for t in autotexts:
        t.set_color(COLORS['text'])
        t.set_fontsize(9)
    ax1.set_title('Data source 비중', fontsize=11, color=COLORS['text'], pad=10)

    # Annotation arrow to MM slice
    ax1.annotate('Target embodiment\nonly 2.4%!\n(400h, 100 homes)',
                 xy=(1.05, 0.55), xytext=(1.5, 0.85),
                 fontsize=10, color=COLORS['orange'], fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=COLORS['orange'], lw=1.3),
                 ha='left')

    # Right panel: bar of what each source contributes
    contributions = [
        ('MM', 'Target embodiment expertise', COLORS['orange']),
        ('ME', 'Diverse environments', COLORS['purple_light']),
        ('CE', 'Cross-embodiment manipulation', COLORS['green_light']),
        ('HL', 'High-level subtask reasoning', '#f5d4a0'),
        ('WD', 'OOD object recognition\n(VQA, captioning)', COLORS['red_light']),
        ('VI', 'Verbal instruction\n(post-train only, 11%)', '#dccaeb'),
    ]
    y_positions = np.arange(len(contributions))[::-1]
    for y, (src, contrib, color) in zip(y_positions, contributions):
        ax2.barh(y, 1.0, color=color, edgecolor=COLORS['gray'], linewidth=1.0)
        ax2.text(0.02, y, src,
                 ha='left', va='center', fontsize=11, fontweight='bold',
                 color=COLORS['text'])
        ax2.text(0.15, y, contrib,
                 ha='left', va='center', fontsize=10, color=COLORS['text'])

    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(-0.6, len(contributions) - 0.4)
    ax2.set_xticks([])
    ax2.set_yticks([])
    clean_axes(ax2)
    ax2.set_title('각 source가 generalization의 다른 axis에 기여',
                  fontsize=11, color=COLORS['text'], pad=10)

    plt.subplots_adjust(top=0.90, bottom=0.05, left=0.03, right=0.98, wspace=0.05)
    save_both(fig, 'pi05_data_mixture')
    plt.close(fig)


# =====================================================
# Figure 4: Adaptive RMSNorm vs MLP fusion (pi0.5 vs pi0)
# =====================================================
def fig_tau_injection_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor('white')
    fig.suptitle('τ injection 비교: π0 (MLP fusion) vs π0.5 (Adaptive RMSNorm)',
                 fontsize=14, fontweight='bold', y=0.99)

    # ============ π0: MLP fusion at input ============
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 9)
    ax1.set_xticks([])
    ax1.set_yticks([])
    clean_axes(ax1)
    ax1.set_title('π0: τ를 INPUT embedding에 fuse',
                  fontsize=12, color=COLORS['purple'], pad=10)

    # τ source
    tau_box = FancyBboxPatch((0.5, 7.5), 1.6, 0.8,
                             boxstyle='round,pad=0.05,rounding_size=0.05',
                             facecolor='#fff4e0', edgecolor=COLORS['orange'])
    ax1.add_patch(tau_box)
    ax1.text(1.3, 7.9, 'τ', ha='center', fontsize=12, color=COLORS['orange'])
    # action
    a_box = FancyBboxPatch((3.5, 7.5), 2.0, 0.8,
                           boxstyle='round,pad=0.05,rounding_size=0.05',
                           facecolor=COLORS['green_light'], edgecolor=COLORS['green'])
    ax1.add_patch(a_box)
    ax1.text(4.5, 7.9, 'a$^τ$', ha='center', fontsize=12, color=COLORS['green'])
    # MLP fusion
    mlp_box = FancyBboxPatch((1.8, 5.5), 4.0, 1.4,
                             boxstyle='round,pad=0.05,rounding_size=0.06',
                             facecolor='#fff4e0', alpha=0.6,
                             edgecolor=COLORS['orange'], linewidth=1.5)
    ax1.add_patch(mlp_box)
    ax1.text(3.8, 6.4, 'MLP fusion', fontsize=11, fontweight='bold',
             ha='center', color=COLORS['orange'])
    ax1.text(3.8, 5.85,
             r'$W_3 \cdot$ swish$(W_2 \cdot $concat$(W_1 \cdot a, \phi(\tau)))$',
             ha='center', fontsize=9, color=COLORS['text'])

    # Arrows from tau, a -> MLP
    for x0 in [1.3, 4.5]:
        ax1.add_patch(FancyArrowPatch((x0, 7.45), (3.8, 6.95),
                                      arrowstyle='-|>', mutation_scale=14,
                                      color=COLORS['gray'], linewidth=1.2))

    # Layers (transformer)
    for k in range(4):
        y = 0.8 + k * 1.0
        layer = Rectangle((1.5, y), 4.6, 0.7,
                          facecolor=COLORS['purple_light'], alpha=0.4,
                          edgecolor=COLORS['purple'])
        ax1.add_patch(layer)
        ax1.text(3.8, y + 0.35, f'Transformer layer {k + 1}',
                 ha='center', va='center', fontsize=10, color=COLORS['text'])

    # Arrow from MLP -> first layer
    ax1.add_patch(FancyArrowPatch((3.8, 5.45), (3.8, 4.55),
                                  arrowstyle='-|>', mutation_scale=14,
                                  color=COLORS['gray'], linewidth=1.4))
    # Vertical arrow between layers (just stack)
    ax1.text(7.0, 2.5, 'τ 정보가\nINPUT에 한 번만\n주입됨',
             ha='center', fontsize=10, color=COLORS['purple_light'].replace('#a8a3d4', '#7a72b5'),
             color2=None) if False else \
        ax1.text(7.0, 2.5, 'τ 정보가\nINPUT에 한 번만\n주입됨',
                 ha='center', fontsize=10, color='#7a72b5', fontweight='bold')

    # ============ π0.5: Adaptive RMSNorm at each layer ============
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 9)
    ax2.set_xticks([])
    ax2.set_yticks([])
    clean_axes(ax2)
    ax2.set_title('π0.5: τ를 매 layer의 RMSNorm parameter로 modulate',
                  fontsize=12, color=COLORS['green'], pad=10)

    # τ source
    tau_box2 = FancyBboxPatch((0.3, 7.5), 1.6, 0.8,
                              boxstyle='round,pad=0.05,rounding_size=0.05',
                              facecolor='#fff4e0', edgecolor=COLORS['orange'])
    ax2.add_patch(tau_box2)
    ax2.text(1.1, 7.9, 'τ', ha='center', fontsize=12, color=COLORS['orange'])

    # τ MLP (separate)
    tau_mlp = FancyBboxPatch((0.3, 5.5), 1.6, 1.4,
                             boxstyle='round,pad=0.05,rounding_size=0.05',
                             facecolor='#fff4e0', alpha=0.6,
                             edgecolor=COLORS['orange'])
    ax2.add_patch(tau_mlp)
    ax2.text(1.1, 6.4, 'τ MLP', fontsize=10, ha='center', fontweight='bold',
             color=COLORS['orange'])
    ax2.text(1.1, 5.9, '(separate)', fontsize=9, ha='center',
             color=COLORS['text_muted'], style='italic')

    # tau -> mlp
    ax2.add_patch(FancyArrowPatch((1.1, 7.45), (1.1, 6.95),
                                  arrowstyle='-|>', mutation_scale=14,
                                  color=COLORS['gray'], linewidth=1.2))

    # action
    a_box2 = FancyBboxPatch((4.5, 7.5), 2.0, 0.8,
                            boxstyle='round,pad=0.05,rounding_size=0.05',
                            facecolor=COLORS['green_light'], edgecolor=COLORS['green'])
    ax2.add_patch(a_box2)
    ax2.text(5.5, 7.9, 'a$^τ$', ha='center', fontsize=12, color=COLORS['green'])

    # Layers w/ AdaLN
    for k in range(4):
        y = 0.8 + k * 1.4
        layer = Rectangle((3.0, y), 4.5, 1.0,
                          facecolor=COLORS['purple_light'], alpha=0.4,
                          edgecolor=COLORS['purple'])
        ax2.add_patch(layer)
        ax2.text(5.25, y + 0.65, f'Transformer layer {k + 1}',
                 ha='center', fontsize=10, color=COLORS['text'])
        ax2.text(5.25, y + 0.2, '(adaptive RMSNorm)',
                 ha='center', fontsize=8, color=COLORS['text_muted'])

        # τ embed feed-in (arrow from MLP region)
        ax2.add_patch(FancyArrowPatch((2.0, 6.2), (3.0, y + 0.5),
                                      arrowstyle='-|>', mutation_scale=12,
                                      color=COLORS['orange'], linewidth=1.1, alpha=0.7,
                                      connectionstyle='arc3,rad=-0.15'))

    # action -> first layer
    ax2.add_patch(FancyArrowPatch((5.5, 7.45), (5.5, 6.0),
                                  arrowstyle='-|>', mutation_scale=14,
                                  color=COLORS['gray'], linewidth=1.4))

    ax2.text(8.5, 4.0, 'τ 정보가\n매 layer 마다\n주입됨',
             ha='center', fontsize=10, color=COLORS['green'], fontweight='bold')

    fig.text(0.5, 0.04,
             'π0.5는 DiT (Diffusion Transformer) 표준 기법으로 회귀 — fine-grained 시간 정보 흐름',
             ha='center', fontsize=11, color=COLORS['text_muted'], style='italic')

    plt.subplots_adjust(top=0.91, bottom=0.10, left=0.03, right=0.98, wspace=0.1)
    save_both(fig, 'pi05_tau_injection_comparison')
    plt.close(fig)


if __name__ == '__main__':
    fig_hierarchical_inference()
    fig_hybrid_fast_flow()
    fig_data_mixture()
    fig_tau_injection_comparison()
    print('All pi0.5 figures generated.')
