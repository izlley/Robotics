"""Sample SVG figure: π0의 최종 τ 분포 (shifted Beta distribution)"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import beta as beta_dist
import matplotlib.font_manager as fm

# Korean font setup
for fp in ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
           '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
           '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc']:
    try:
        fm.fontManager.addfont(fp)
        prop = fm.FontProperties(fname=fp)
        matplotlib.rcParams['font.family'] = prop.get_name()
        print(f"Using font: {prop.get_name()}")
        break
    except Exception as e:
        continue
else:
    print("WARNING: no Korean font found")

matplotlib.rcParams['axes.unicode_minus'] = False

# 파라미터
s = 0.999
alpha, beta = 1.5, 1.0

# τ 값
tau = np.linspace(0, s, 500)
u = (s - tau) / s
p = beta_dist.pdf(u, alpha, beta) / s

# 그림
fig, ax = plt.subplots(figsize=(11, 6.5))
fig.patch.set_facecolor('white')

ax.fill_between(tau, 0, p, color='#a8a3d4', alpha=0.5)
ax.plot(tau, p, color='#5b56b5', linewidth=2.0)

ax.plot(0, 1.5, marker='o', markersize=10, markerfacecolor='white',
        markeredgecolor='#c8694f', markeredgewidth=2.0, zorder=10)
ax.annotate('peak (τ=0)\n1.5',
            xy=(0, 1.5), xytext=(0.03, 1.45),
            fontsize=12, color='#333',
            ha='left', va='top')

ax.axvline(x=s, ymin=0, ymax=0.95, color='#8b3a3a',
           linestyle='--', linewidth=1.8, alpha=0.8)
ax.annotate('τ = 0.999\n절단',
            xy=(s, 1.4), xytext=(s - 0.02, 1.45),
            fontsize=11, color='#8b3a3a', ha='right', va='top')

ax.set_xlim(-0.02, 1.05)
ax.set_ylim(0, 1.7)
ax.set_xlabel('τ', fontsize=14)
ax.set_ylabel('p(τ)', fontsize=14, rotation=0, labelpad=18, va='center')
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_yticks([0, 1.5])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=11)

fig.suptitle('π$_0$의 최종 τ 분포', fontsize=15, fontweight='bold', y=0.97)
ax.set_title('p(τ) = Beta((0.999 − τ)/0.999 ; 1.5, 1)',
             fontsize=12, color='#333', pad=14)

quarters = [(0.0, 0.25, '≈ 40%'),
            (0.25, 0.5, '≈ 30%'),
            (0.5, 0.75, '≈ 20%'),
            (0.75, 1.0, '≈ 10%')]

box_y = -0.32
for x0, x1, label in quarters:
    rect = Rectangle((x0, box_y), x1 - x0, 0.13,
                     facecolor='#dde9e1', edgecolor='#a3c4b0',
                     transform=ax.get_xaxis_transform(), clip_on=False)
    ax.add_patch(rect)
    ax.text((x0 + x1) / 2, box_y + 0.065, label,
            transform=ax.get_xaxis_transform(),
            ha='center', va='center', fontsize=11)

ax.text(0.5, -0.50, 'τ ∈ [0, 0.25] 구간이 가장 자주 샘플링됨 (어려운 영역에 집중)',
        transform=ax.get_xaxis_transform(),
        ha='center', va='center', fontsize=11, style='italic', color='#444')

plt.tight_layout(rect=[0, 0.05, 1, 0.95])

plt.savefig('/workspace/izlley/sllm/Robotics/reports/papers/figures/pi0_tau_distribution.svg',
            format='svg', bbox_inches='tight', facecolor='white')
plt.savefig('/workspace/izlley/sllm/Robotics/reports/papers/figures/pi0_tau_distribution.png',
            format='png', dpi=150, bbox_inches='tight', facecolor='white')
print('Saved: pi0_tau_distribution.svg + .png')
