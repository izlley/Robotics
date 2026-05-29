"""Shared style helper for SVG figures in reports/papers/figures/.

Usage:
    from _style import setup, COLORS, save_both
    setup()
    fig, ax = ...
    save_both(fig, 'my_diagram')
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

FIGURE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Color palette (consistent across all figures)
COLORS = {
    'purple': '#5b56b5',
    'purple_light': '#a8a3d4',
    'purple_fill': '#c0bbe0',
    'green': '#3da084',
    'green_light': '#d8eee0',
    'red': '#8b3a3a',
    'red_light': '#e8c5c5',
    'orange': '#c8694f',
    'orange_light': '#f0d4c4',
    'gray': '#7a7a7a',
    'gray_light': '#e7e6e2',
    'box_fill': '#dde9e1',
    'box_edge': '#a3c4b0',
    'text': '#333',
    'text_muted': '#666',
}


def setup():
    """Setup matplotlib with Korean font + clean style."""
    # Try Korean fonts in priority order
    for fp in [
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    ]:
        try:
            fm.fontManager.addfont(fp)
            prop = fm.FontProperties(fname=fp)
            matplotlib.rcParams['font.family'] = prop.get_name()
            break
        except Exception:
            continue
    matplotlib.rcParams['axes.unicode_minus'] = False
    # Use ASCII '-' for minus when needed via 'mathtext.default' fallback
    matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'


def save_both(fig, name):
    """Save SVG (vector) and PNG (raster) versions to reports/papers/figures/."""
    svg_path = os.path.join(FIGURE_DIR, f'{name}.svg')
    png_path = os.path.join(FIGURE_DIR, f'{name}.png')
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    fig.savefig(png_path, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {name}.svg + .png')


def clean_axes(ax, keep=()):
    """Hide spines except those in keep tuple."""
    for spine in ['top', 'right', 'bottom', 'left']:
        if spine not in keep:
            ax.spines[spine].set_visible(False)


def add_box(ax, x, y, w, h, label, fc=None, ec=None, text_color=None, fontsize=11):
    """Convenience: add a rounded box with centered label."""
    from matplotlib.patches import FancyBboxPatch
    fc = fc or COLORS['gray_light']
    ec = ec or COLORS['gray']
    text_color = text_color or COLORS['text']
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle='round,pad=0.02,rounding_size=0.05',
                         facecolor=fc, edgecolor=ec, linewidth=1.3)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label,
            ha='center', va='center', fontsize=fontsize, color=text_color)
