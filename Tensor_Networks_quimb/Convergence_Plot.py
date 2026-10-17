# Overlays every (max_bond, timestep) run in an output folder, one page per J, to check convergence
# Run with: python Convergence_Plot.py Output/L7_N8_T100/Corrected
# The original L7_N8_T100 data was double-normalised, so point this at its Corrected/ folder; data from the fixed Functions.py can be used directly

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

Run_Dir = sys.argv[1]
Run = os.path.basename(os.path.normpath(Run_Dir))
if Run == 'Corrected':
    Run = os.path.basename(os.path.dirname(os.path.normpath(Run_Dir))) # Name the plot after the run, not the Corrected/ subfolder

CHI_COLORS = {32: '#2a78d6', 64: '#eb6834', 128: '#1baf7a', 256: '#eda100'}   # Categorical slots 1-4, colour follows bond dimension
DT_STYLES = {0.1: (0, (2, 2)), 0.05: '-', 0.025: (0, (6, 2))}                  # Line style follows timestep
TRUNC_LIMIT = 1e-2 # Times after the best run's cumulative truncation error passes this are shaded
SITE = 3           # Site plotted (middle site of L=7)

runs = {}
for f in sorted(glob.glob(os.path.join(Run_Dir, '*.npz'))):
    d = np.load(f, allow_pickle=False)
    chi, dt = int(d['max_bond']), float(d['Timestep'])
    runs[(chi, dt)] = {round(float(J), 2): (d['times'], d['occupation'][i], d['magnetization'][i], d['truncation_error'][i]) for i, J in enumerate(d['J'])}

Js = sorted({J for r in runs.values() for J in r})
Conv_Path = os.path.join(Run_Dir, f'Convergence_{Run}.pdf')
with PdfPages(Conv_Path) as pdf:
    for J in Js:
        fig, (ax_n, ax_s, ax_t) = plt.subplots(3, 1, figsize=(10, 9), sharex=True, layout='constrained')
        # Trust limit from the most accurate run available for this J (largest max_bond, then smallest timestep)
        best = max((k for k in runs if J in runs[k]), key=lambda k: (k[0], -k[1]))
        times, trunc = runs[best][J][0], runs[best][J][3]
        over = np.where(trunc > TRUNC_LIMIT)[0]
        t_star = times[over[0]] if len(over) else None
        for (chi, dt), r in sorted(runs.items()):
            if J not in r: continue
            times, occ, mag, tr = r[J]
            kw = dict(color=CHI_COLORS.get(chi, '0.3'), linestyle=DT_STYLES.get(dt, '-'), lw=1.5, label=f'χ={chi}, dt={dt}')
            ax_n.plot(times, occ[:, SITE], **kw)
            ax_s.plot(times, mag[:, SITE], **kw)
            ax_t.semilogy(times, np.maximum(tr, 1e-8), **kw)
        for ax in (ax_n, ax_s, ax_t):
            if t_star is not None:
                ax.axvspan(t_star, times[-1], color='0.5', alpha=0.12, lw=0)
            ax.grid(alpha=0.25, lw=0.5)
            for side in ('top', 'right'): ax.spines[side].set_visible(False)
        ax_t.axhline(TRUNC_LIMIT, color='0.4', lw=0.8, linestyle=':')
        ax_n.set_ylabel(f'<n_{SITE}>'); ax_s.set_ylabel(f'<sz_{SITE}>'); ax_t.set_ylabel('1 - <psi|psi>'); ax_t.set_xlabel('t')
        handles, labels = ax_n.get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', fontsize=8, ncol=4, frameon=False)
        shade = f'shaded: t > {t_star:g}, where χ={best[0]}, dt={best[1]} truncation > {TRUNC_LIMIT:g}' if t_star is not None else f'truncation < {TRUNC_LIMIT:g} for all t (χ={best[0]}, dt={best[1]})'
        fig.suptitle(f'{Run}  J={J}  (site {SITE})\n{shade}', fontsize=10)
        pdf.savefig(fig); plt.close(fig)
print(f'Saved {Conv_Path}')
