"""Vector figures for the TAFFC submission. Numbers are copied from results/*.csv (28 Oct - 1 Nov 2026 runs)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np, os

OUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42})
GREY, BLUE, DARK = '#9a9a9a', '#1f77b4', '#222222'

# ---------------- Fig. 1: architecture ----------------
fig, ax = plt.subplots(figsize=(7.0, 3.3)); ax.set_xlim(0, 100); ax.set_ylim(-1, 50); ax.axis('off')
def box(x, y, w, h, text, fc='#f4f4f4', ec=DARK, fs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.2,rounding_size=1.0', fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=DARK, linespacing=1.25)
def arrow(x1, y1, x2, y2, ls='-', c=DARK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=8, lw=0.9, color=c, linestyle=ls, shrinkA=0, shrinkB=0))
box(0.5, 26, 21, 12, 'WavLM-Large (frozen)\n25 layers, mean-pooled,\nlearned layer softmax', fc='#e8eef7', fs=6.2)
box(0.5, 9, 21, 11, 'RoBERTa-base (frozen)\nmasked mean-pool', fc='#e8eef7', fs=6.2)
box(24, 18, 14, 12, 'z-score,\nPCA-initialised\nprojection $W_p$\n(H = 128)', fs=6.2)
arrow(21.5, 32, 24, 27); arrow(21.5, 14.5, 24, 21)
box(21, 37, 41, 11, 'Hardness $h(t)$: normalised label entropy of the\n$K$ nearest training-fold neighbours of $f(t)$\n(own utterance, dialogue, speaker excluded; no target label)', fc='#fff3dc', fs=5.9)
box(42, 18, 14, 12, 'FiLM gate\n$(1+\\gamma h)\\odot z$\n$+\\,\\beta h$', fs=6.2)
arrow(38, 24, 42, 24); arrow(49, 37, 49, 30.5)
box(60, 11, 19, 25, 'Context window\n$t-8,\\ldots,t$\nmasked to $k(t)$ slots\n+ position emb.\n+ same-speaker flag\n\nattention (4 heads)\nor BiGRU', fc='#f0f7ef', fs=6.2)
arrow(56, 24, 60, 24)
box(83, 28.5, 16, 10, 'MLP classifier\n$p(y\\mid\\mathrm{window})$', fs=6.2)
arrow(79, 30, 83, 33)
box(83, 8, 16, 14, '$k$-predictor $g(z,h)$\n6 classes:\n$k\\in\\{0,1,2,4,8\\}$\nor unresolvable', fc='#fff3dc', fs=6.3)
arrow(49, 18, 49, 14); arrow(49, 14, 83, 14, ls='--', c='#666666'); arrow(58, 42, 91, 42, ls='--', c='#666666'); arrow(91, 42, 91, 38.5, ls='--', c='#666666'); arrow(91, 28, 91, 22, ls='--', c='#666666')
ax.annotate('', xy=(79, 12), xytext=(83, 10), arrowprops=dict(arrowstyle='-|>', ls=':', lw=0.9, color='#666666'))
ax.text(68, 4.5, 'dotted: predicted $k(t)$ sets the window at inference\ndashed: inputs to $g$; $g$ is supervised by the oracle $k^*$ (inner CV, training folds only)', ha='center', va='center', fontsize=6.0, color='#555555')
ax.text(1, 1.5, 'encoders frozen; 0.32 M trainable parameters below them', fontsize=6.0, color='#555555', ha='left')
fig.savefig(f'{OUT}/fig1_architecture.pdf', bbox_inches='tight'); plt.close(fig)

# ---------------- Fig. 3: W-F1 vs mean context ----------------
fixed = [(0, 58.80), (1, 62.35), (2, 63.51), (4, 65.83), (8, 67.17)]
agn = [(0, 58.80), (8, 66.73)]
hard = [(4.95, 64.71, r'$\tau_k$=0.5'), (5.22, 65.07, r'$\tau_k$=0.7'), (6.55, 66.26, r'$\tau_k$=0.9')]
rnd = [(4.95, 65.87), (5.22, 66.46), (6.55, 66.50)]
fig, ax = plt.subplots(figsize=(3.5, 2.6))
ax.plot(*zip(*fixed), '-o', color=GREY, ms=4, lw=1.4, label='fixed $k$, classifier trained at that $k$')
for k, v in fixed: ax.annotate(f'{v:.1f}', (k, v), textcoords='offset points', xytext=(0, 5 if k else -11), ha='center', fontsize=6.5, color='#555555')
ax.plot(*zip(*agn), '--', color=GREY, lw=1.1, label='window-agnostic classifier ($k$=0 and 8)')
ax.scatter([r[0] for r in rnd], [r[1] for r in rnd], s=38, facecolors='none', edgecolors=GREY, lw=1.4, label=r'random-$k$ at the same $\bar k$ (inference)', zorder=4)
ax.scatter([h[0] for h in hard], [h[1] for h in hard], s=38, color=BLUE, label=r'hardness-$k$ (inference)', zorder=5)
ax.scatter([6.55], [65.47], s=50, marker='D', facecolors='none', edgecolors=BLUE, lw=1.4, label=r'hardness-$k$, retrained (65.5$\pm$1.7)', zorder=5)
ax.scatter([6.55], [64.76], s=50, marker='s', facecolors='none', edgecolors=GREY, lw=1.4, label=r'random-$k$, retrained (64.8$\pm$2.1)', zorder=5)
ax.set_xlabel(r'mean context length $\bar k$ (preceding utterances read)'); ax.set_ylabel('weighted-F1 (IEMOCAP, Ses05)')
ax.set_xticks([0, 1, 2, 4, 8]); ax.set_ylim(56, 69.5); ax.grid(axis='y', color='#e5e5e5', lw=0.6)
ax.legend(fontsize=5.6, loc='lower right', frameon=False)
fig.savefig(f'{OUT}/fig3_wf1_vs_kbar.pdf', bbox_inches='tight'); plt.close(fig)

# ---------------- Fig. 4: gain by hardness quartile ----------------
iem = dict(q=['Q1\n(easiest)', 'Q2', 'Q3', 'Q4\n(hardest)'], h=[0.28, 0.49, 0.58, 0.74], k0=[67.64, 57.63, 57.73, 52.17], k8=[72.46, 67.40, 65.95, 63.32])
meld = dict(q=['Q1\n(easiest)', 'Q2', 'Q3', 'Q4\n(hardest)'], h=[0.39, 0.57, 0.66, 0.79], k0=[68.54, 63.78, 60.12, 57.59], k8=[67.68, 63.66, 60.14, 57.94])
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
for ax, d, title in zip(axes, [iem, meld], ['IEMOCAP (speech + text)', 'MELD (speech + text)']):
    x = np.arange(4); w = 0.36
    ax.bar(x - w / 2, d['k0'], w, color=GREY, label='$k$ = 0 (no context)')
    ax.bar(x + w / 2, d['k8'], w, color=BLUE, alpha=0.9, label='$k$ = 8')
    for i in range(4):
        g = d['k8'][i] - d['k0'][i]
        ax.text(x[i], max(d['k0'][i], d['k8'][i]) + 0.8, f'{np.round(g + 1e-6, 1):+.1f}', ha='center', fontsize=7.5, fontweight='bold' if abs(g) > 2 else 'normal')
    ax.set_xticks(x); ax.set_xticklabels([f'{q}\n$h$={hh:.2f}' for q, hh in zip(d['q'], d['h'])], fontsize=6.8)
    ax.set_ylim(40, 80); ax.set_title(title, fontsize=8.5); ax.grid(axis='y', color='#e5e5e5', lw=0.6)
axes[0].set_ylabel('weighted-F1 (test, 5 seeds pooled)'); axes[0].legend(fontsize=6.5, frameon=False, loc='upper right')
fig.savefig(f'{OUT}/fig4_gain_by_quartile.pdf', bbox_inches='tight'); plt.close(fig)
print('figures written to', OUT)
