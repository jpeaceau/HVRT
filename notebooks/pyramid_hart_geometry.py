"""
PyramidHART — multi-scale residual geometry figure for LaTeX paper.

Architecture
------------
A residual pyramid where each level fits HART partitioning to the
previous level's residuals:

    X_0 = X                           (original data)
    L0 :  HART(X_0, K=4)  → P_0      (partition → prototype map)
    ε_0 = X_0 − P_0(X_0)             (level-0 residuals)

    L1 :  HART(ε_0, K=8)  → P_1
    ε_1 = ε_0 − P_1(ε_0)             (level-1 residuals)

    L2 :  HART(ε_1, K=12) → P_2
    ε_2 = ε_1 − P_2(ε_1)             (level-2 residuals)

Reconstruction at depth d:
    X̂_d = P_0(X_0) + P_1(ε_0) + ... + P_d(ε_{d-1})

Figure layout  (2 rows × 4 columns)
------------------------------------
Row 0 | partition geometry at each level (scatter + coloured background)
Row 1 | original-space reconstruction error (colour = ‖X − X̂_k‖₂)
      | rightmost panel: error bar chart

Usage
-----
    python notebooks/pyramid_hart_geometry.py

Output
------
    notebooks/pyramid_hart_geometry.pdf
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_moons

# ── Publication-style rcParams ────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':          9,
    'axes.titlesize':     9,
    'axes.labelsize':     8,
    'xtick.labelsize':    7,
    'ytick.labelsize':    7,
    'axes.linewidth':     0.75,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'xtick.major.width':  0.6,
    'ytick.major.width':  0.6,
    'figure.dpi':         150,
})

# ── Pyramid hyperparameters ───────────────────────────────────────────────────
N_PARTITIONS = [4, 8, 12]   # K increases with depth
SEED         = 42
_CONSISTENCY = 1.4826        # MAD → σ consistency factor (Gaussian)

# Colour scheme: one hue per level (matches HVRT/HART benchmark palette)
LEVEL_DARK  = ['#1565C0', '#E65100', '#2E7D32']   # scatter / bars
LEVEL_CMAP  = ['Blues',   'Oranges', 'Greens']     # partition background
LEVEL_LABEL = [
    r'Level 0  ($K=4$)',
    r'Level 1  ($K=8$)',
    r'Level 2  ($K=12$)',
]
_BG_ALPHA = 0.20   # partition-background translucency


# ── MAD normalisation ─────────────────────────────────────────────────────────
def _mad_scale(X):
    """Return (X_z, center, scale) using median + 1.4826·MAD."""
    center = np.median(X, axis=0)
    mad    = np.median(np.abs(X - center), axis=0)
    scale  = np.where(mad > 0, _CONSISTENCY * mad, 1.0)
    return (X - center) / scale, center, scale


# ── FastHART synthetic target  O(d) ──────────────────────────────────────────
def _hart_target(X_z):
    """Sum of |z_j|  — FastHART x-component."""
    return np.abs(X_z).sum(axis=1)


# ── Single HART partition level ───────────────────────────────────────────────
def _fit_level(X, n_parts, seed):
    """
    Fit one HART level to X.

    Returns
    -------
    part_ids   (n,)    compact 0-indexed partition assignments
    prototypes (n, d)  per-point partition median (prototype)
    residuals  (n, d)  X − prototype
    tree               fitted DecisionTreeRegressor
    center, scale      MAD parameters (for grid prediction)
    n_actual           number of leaf nodes actually used
    """
    X_z, center, scale = _mad_scale(X)
    target = _hart_target(X_z)

    min_leaf = max(2, len(X) // (n_parts * 8))
    tree = DecisionTreeRegressor(
        criterion     = 'absolute_error',
        max_leaf_nodes= n_parts,
        min_samples_leaf = min_leaf,
        random_state  = seed,
    )
    tree.fit(X_z, target)
    raw_ids = tree.apply(X_z.astype(np.float32))

    # Remap raw leaf IDs → compact 0 … K-1
    uid     = np.unique(raw_ids)
    remap   = {v: i for i, v in enumerate(uid)}
    part_ids = np.array([remap[r] for r in raw_ids])

    # Partition prototypes: column-wise median in *original* X space
    prototypes = np.empty_like(X)
    for pid in range(len(uid)):
        mask = part_ids == pid
        prototypes[mask] = np.median(X[mask], axis=0)

    return part_ids, prototypes, X - prototypes, tree, center, scale, len(uid)


# ── Full PyramidHART ──────────────────────────────────────────────────────────
def fit_pyramid(X):
    """Fit PyramidHART; return list of level dicts."""
    levels, X_cur = [], X.copy()
    for k, n_parts in enumerate(N_PARTITIONS):
        pids, protos, resids, tree, center, scale, n_act = \
            _fit_level(X_cur, n_parts, seed=SEED + k)
        levels.append(dict(
            data       = X_cur,
            part_ids   = pids,
            prototypes = protos,
            residuals  = resids,
            tree       = tree,
            center     = center,
            scale      = scale,
            n_actual   = n_act,
        ))
        X_cur = resids
    return levels


# ── Grid partition prediction (for coloured background) ──────────────────────
def _partition_grid(lev, n_grid=260):
    data   = lev['data']
    margin = 0.30
    xl, xh = data[:, 0].min() - margin, data[:, 0].max() + margin
    yl, yh = data[:, 1].min() - margin, data[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(xl, xh, n_grid),
                         np.linspace(yl, yh, n_grid))
    pts    = np.c_[xx.ravel(), yy.ravel()]
    pts_z  = (pts - lev['center']) / lev['scale']
    raw    = lev['tree'].apply(pts_z.astype(np.float32))
    # Remap to compact IDs (same remap as in _fit_level)
    uid    = np.unique(lev['part_ids'])
    # raw leaf IDs → compact via prototype assignment match
    # Easier: remap raw directly
    uid_raw = np.unique(raw)
    remap   = {v: i for i, v in enumerate(uid_raw)}
    zz      = np.vectorize(remap.get)(raw).reshape(xx.shape)
    return xx, yy, zz, len(uid_raw)


# ── Discrete ListedColormap ───────────────────────────────────────────────────
def _disc_cmap(n, cmap_name, lo=0.30, hi=0.90):
    src  = plt.get_cmap(cmap_name)
    cols = [src(lo + (hi - lo) * i / max(1, n - 1)) for i in range(n)]
    return ListedColormap(cols)


# ── Cumulative reconstruction ─────────────────────────────────────────────────
def _cumulative_recons(levels, X_orig):
    """
    Returns [(X_hat_k, per_point_error_k), ...] for k = 0, 1, 2.

    X_hat_k  = sum of prototypes from levels 0 … k
    error_k  = ‖X_orig − X_hat_k‖₂  per point
    """
    recons, acc = [], np.zeros_like(X_orig)
    for lev in levels:
        acc = acc + lev['prototypes']
        recons.append((acc.copy(),
                       np.linalg.norm(X_orig - acc, axis=1)))
    return recons


# ── Figure ────────────────────────────────────────────────────────────────────
def draw_figure(X, levels, recons, out_path):
    n_lev = len(levels)

    fig = plt.figure(figsize=(14.0, 6.2))
    gs  = GridSpec(2, n_lev + 1, figure=fig,
                   left=0.05, right=0.97,
                   top=0.88, bottom=0.10,
                   wspace=0.10, hspace=0.48)

    # ── Top row: partition geometry ───────────────────────────────────────────
    for k, lev in enumerate(levels):
        ax   = fig.add_subplot(gs[0, k])
        cmap = _disc_cmap(lev['n_actual'], LEVEL_CMAP[k])

        xx, yy, zz, n_grid_parts = _partition_grid(lev)
        # Background: coloured partition regions
        ax.pcolormesh(xx, yy, zz, cmap=cmap, shading='auto',
                      alpha=_BG_ALPHA, zorder=0, rasterized=True)

        # Data scatter coloured by partition
        c_vals = cmap(lev['part_ids'] / max(1, lev['n_actual'] - 1))
        ax.scatter(lev['data'][:, 0], lev['data'][:, 1],
                   c=c_vals, s=11, lw=0.15,
                   edgecolors='white', alpha=0.85, zorder=3)

        # Partition prototypes as black crosses
        proto_pos = np.array([
            np.median(lev['data'][lev['part_ids'] == pid], axis=0)
            for pid in range(lev['n_actual'])
        ])
        ax.scatter(proto_pos[:, 0], proto_pos[:, 1],
                   marker='+', s=100, lw=1.6,
                   c='black', zorder=5,
                   label='Prototype' if k == 0 else None)

        if k == 0:
            data_lbl = r'$\mathbf{X}$  (original)'
        else:
            data_lbl = rf'$\varepsilon_{{{k-1}}}$  (residuals)'
        ax.set_title(f'{LEVEL_LABEL[k]}\n{data_lbl}', fontsize=8.5, pad=3)
        ax.set_xlabel(r'$x_1$' if k == 0 else
                      rf'$\varepsilon_{{{k-1},\,1}}$', fontsize=8)
        ax.set_ylabel(r'$x_2$' if k == 0 else '', fontsize=8)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=6.5)

        if k == 0:
            ax.legend(fontsize=6.5, handletextpad=0.3,
                      loc='lower right', framealpha=0.85,
                      edgecolor='0.8', markerscale=0.9)

    # ── Bottom row: reconstruction error in original-X space ─────────────────
    vmax = recons[0][1].max()   # fix colour scale to depth-0 (worst) error

    for k, (X_hat, err) in enumerate(recons):
        ax = fig.add_subplot(gs[1, k])
        sc = ax.scatter(X[:, 0], X[:, 1],
                        c=err, cmap='magma_r',
                        vmin=0, vmax=vmax,
                        s=11, lw=0, alpha=0.85, zorder=3)
        if k == 0:
            cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03,
                                orientation='vertical')
            cbar.set_label(r'$\|\mathbf{X} - \hat{\mathbf{X}}_k\|_2$',
                           fontsize=7)
            cbar.ax.tick_params(labelsize=6)

        if k == 0:
            recon_lbl = r'$\hat{\mathbf{X}}_0 = P_0(\mathbf{X})$'
        else:
            recon_lbl = (rf'$\hat{{\mathbf{{X}}}}_{k} = '
                         rf'\hat{{\mathbf{{X}}}}_{{{k-1}}} + '
                         rf'P_{k}(\varepsilon_{{{k-1}}})$')
        ax.set_title(f'Depth {k}  reconstruction\n{recon_lbl}',
                     fontsize=8.0, pad=3)
        ax.set_xlabel(r'$x_1$', fontsize=8)
        ax.set_ylabel(r'$x_2$' if k == 0 else '', fontsize=8)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=6.5)

    # ── Right column, top: original vs full reconstruction overlay ────────────
    ax_over = fig.add_subplot(gs[0, n_lev])
    X_final = recons[-1][0]
    ax_over.scatter(X[:, 0], X[:, 1],
                    c=LEVEL_DARK[0], s=9, lw=0, alpha=0.30,
                    label=r'$\mathbf{X}$', zorder=2)
    ax_over.scatter(X_final[:, 0], X_final[:, 1],
                    c=LEVEL_DARK[1], s=11, marker='x', lw=0.9, alpha=0.55,
                    label=r'$\hat{\mathbf{X}}_2$', zorder=3)
    ax_over.set_title('Original vs.\nfull reconstruction', fontsize=8.5, pad=3)
    ax_over.set_xlabel(r'$x_1$', fontsize=8)
    ax_over.set_aspect('equal')
    ax_over.tick_params(labelsize=6.5)
    ax_over.legend(fontsize=7, framealpha=0.85, edgecolor='0.8',
                   loc='lower right', handletextpad=0.2)

    # ── Right column, bottom: bar chart of mean error per depth ───────────────
    ax_bar = fig.add_subplot(gs[1, n_lev])
    mean_errs = [float(e.mean()) for _, e in recons]
    bars = ax_bar.bar(
        [f'Depth {k}' for k in range(n_lev)],
        mean_errs,
        color=LEVEL_DARK, width=0.48, alpha=0.88,
    )
    for bar, val in zip(bars, mean_errs):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.003,
                    f'{val:.3f}',
                    ha='center', va='bottom', fontsize=7.5)
    ax_bar.set_ylabel(r'Mean $\|\mathbf{X} - \hat{\mathbf{X}}_k\|_2$',
                      fontsize=7.5)
    ax_bar.set_title('Reconstruction error\nby pyramid depth',
                     fontsize=8.5, pad=3)
    ax_bar.set_ylim(0, max(mean_errs) * 1.28)
    ax_bar.tick_params(axis='x', labelsize=7.5)
    ax_bar.tick_params(axis='y', labelsize=6.5)

    # ── Super-title ────────────────────────────────────────────────────────────
    fig.suptitle(
        'PyramidHART  —  Multi-Scale Residual Decomposition Geometry',
        fontsize=11, fontweight='bold',
    )

    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    rng = np.random.default_rng(SEED)

    # Two-moons dataset: noise=0.07 gives clean crescents with some variation
    X, _ = make_moons(n_samples=500, noise=0.07, random_state=SEED)
    # Standardise: zero mean, unit variance on each axis
    X    = (X - X.mean(0)) / X.std(0)

    levels = fit_pyramid(X)
    recons = _cumulative_recons(levels, X)

    # Print partition counts actually achieved
    for k, lev in enumerate(levels):
        actual = lev['n_actual']
        target = N_PARTITIONS[k]
        print(f'Level {k}: {actual}/{target} partitions  '
              f'| residual RMS = {np.sqrt((lev["residuals"]**2).mean()):.4f}')

    mean_errs = [float(e.mean()) for _, e in recons]
    for k, me in enumerate(mean_errs):
        print(f'Depth {k}: mean reconstruction error = {me:.4f}')

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'pyramid_hart_geometry.pdf')
    draw_figure(X, levels, recons, out_path=out)
