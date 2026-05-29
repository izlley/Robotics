"""Sample SVG figure: 2차원 공간에서 본 직선 경로 (noise -> action)"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
import matplotlib.font_manager as fm

# Korean font
for fp in ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf']:
    try:
        fm.fontManager.addfont(fp)
        prop = fm.FontProperties(fname=fp)
        matplotlib.rcParams['font.family'] = prop.get_name()
        break
    except Exception:
        pass
matplotlib.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor('white')

# ε (noise) at left-bottom and A_t at right-top
eps = np.array([1.5, 2.2])
A_t = np.array([9.5, 6.5])

# Path: τ=0 (eps) to τ=1 (A_t)
taus = [0.0, 0.25, 0.5, 0.75, 1.0]
pts = [eps + tau * (A_t - eps) for tau in taus]

# Background dashed line: full path
ax.plot([eps[0], A_t[0]], [eps[1], A_t[1]],
        color='#7a72c4', linestyle='--', linewidth=1.6, alpha=0.65, zorder=1)

# Velocity arrows between consecutive points
for i in range(len(pts) - 1):
    p1, p2 = pts[i], pts[i + 1]
    # Slightly shorten to leave space at endpoints
    direction = p2 - p1
    length = np.linalg.norm(direction)
    unit = direction / length
    start = p1 + unit * 0.18
    end = p2 - unit * 0.18
    arrow = FancyArrowPatch(start, end,
                            arrowstyle='-|>', mutation_scale=18,
                            color='#3da084', linewidth=2.0, zorder=2)
    ax.add_patch(arrow)

# Endpoint circles (noise = light gray, A_t = light green)
ax.scatter(*eps, s=380, c='#e7e6e2', edgecolors='#8a8884', linewidths=1.5, zorder=3)
ax.scatter(*A_t, s=380, c='#d8eee0', edgecolors='#3da084', linewidths=1.5, zorder=3)

# Intermediate points (purple dots)
for tau, pt in zip(taus[1:-1], pts[1:-1]):
    ax.scatter(*pt, s=110, c='#5b56b5', zorder=4)

# Labels
# ε at τ=0
ax.annotate('ε (τ=0)', xy=eps, xytext=(eps[0] - 0.05, eps[1] - 0.6),
            fontsize=13, color='#555', ha='center')
# A_t at τ=1
ax.annotate('A$_t$ (τ=1)', xy=A_t, xytext=(A_t[0] + 0.1, A_t[1] - 0.5),
            fontsize=13, color='#3da084', ha='left')

# τ labels above intermediate points
tau_labels = ['τ=0.25', 'τ=0.5', 'τ=0.75']
for tau_lbl, pt in zip(tau_labels, pts[1:-1]):
    ax.annotate(tau_lbl, xy=pt, xytext=(pt[0] - 0.7, pt[1] + 0.45),
                fontsize=11, color='#555', ha='left')

# Axes
ax.set_xlim(0, 11.5)
ax.set_ylim(0, 9)
ax.set_xlabel('x', fontsize=13, loc='right')
ax.set_ylabel('y', fontsize=13, rotation=0, va='bottom', ha='left', labelpad=12, y=1.0)
ax.set_xticks([])
ax.set_yticks([])

# Custom axis arrows (origin axes)
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#333')
ax.spines['left'].set_color('#333')

# Titles
fig.suptitle('2차원 공간에서 본 직선 경로', fontsize=15, fontweight='bold', y=0.97)
ax.set_title('노이즈 ε에서 action A$_t$로 가는 직선',
             fontsize=12, color='#444', pad=12)

# Bottom caption
fig.text(0.5, 0.10,
         '모든 지점에서 velocity 벡터(초록)가\n'
         '방향과 크기 모두 동일 = A$_t$ − ε',
         ha='center', fontsize=12, color='#222')
fig.text(0.5, 0.025,
         '직선이라서 미분(접선 방향)이 일정한 상수 벡터',
         ha='center', fontsize=11, color='#666', style='italic')

plt.subplots_adjust(top=0.88, bottom=0.22)

plt.savefig('/workspace/izlley/sllm/Robotics/reports/papers/figures/pi0_linear_path.svg',
            format='svg', bbox_inches='tight', facecolor='white')
plt.savefig('/workspace/izlley/sllm/Robotics/reports/papers/figures/pi0_linear_path.png',
            format='png', dpi=150, bbox_inches='tight', facecolor='white')
print('Saved: pi0_linear_path.svg + .png')
