# Makes all plots from the .npz files produced by Simulation.py, saving the PDFs next to the data
# Run with: python Tensor_Networks_quimb_CREATE/plots.py                 (every folder under Output/)
#      or:  Tensor_Networks_quimb_CREATE/plots.py Tensor_Networks_quimb_CREATE/Output/L2_N8_T100_PBC
# For each folder of .npz files it writes:
#   Bond{χ}_Step{dt}_K{K}_TEBD{o}.pdf  - occupation and magnetization of every site, one row per J
#   Convergence_{Run}.pdf              - all (max_bond, timestep) runs overlaid at the middle site, one page per J (only if there is more than one run to compare)
# Legacy .npz files holding a whole J scan (e.g. Output/L7_N8_T100/Corrected) are also read

import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

parser = argparse.ArgumentParser()
parser.add_argument('path', nargs='?', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Output'), help='Output folder (searched recursively)')
parser.add_argument('--no-runs', action='store_true', help='Skip the per-run occupation/magnetization plots')
parser.add_argument('--no-convergence', action='store_true', help='Skip the convergence plots')
args = parser.parse_args()

CHI_COLORS = {32: '#2a78d6', 64: '#eb6834', 128: '#1baf7a', 256: '#eda100'}   # Categorical slots 1-4, colour follows bond dimension
DT_STYLES = {0.1: (0, (2, 2)), 0.05: '-', 0.025: (0, (6, 2))}                  # Line style follows timestep
TRUNC_LIMIT = 1e-2 # Times after the best run's cumulative truncation error passes this are shaded

#---Loading---#

def load_folder(Run_Dir):
    """Returns (runs, info). runs[(max_bond, Timestep, tebd_order)][J] = dict of arrays cut to the completed samples. info holds the shared model parameters."""
    runs, info = {}, {}
    for f in sorted(glob.glob(os.path.join(Run_Dir, '*.npz'))):
        d = np.load(f, allow_pickle=False)
        key = (int(d['max_bond']), float(d['Timestep']), int(d['tebd_order']))
        info.update(L=int(d['L']), N=int(d['N']), Total_time=d['Total_time'].item(), Steps_per_sample=int(d['Steps_per_sample']), cyclic=bool(d['cyclic']) if 'cyclic' in d else False)
        if d['J'].ndim == 0: # One J per file, possibly still running / crashed
            n = int(d['n_done'])
            runs.setdefault(key, {})[round(float(d['J']), 6)] = dict(times=d['times'][:n], occupation=d['occupation'][:n], magnetization=d['magnetization'][:n],
                                                                     truncation_error=d['truncation_error'][:n], run_time=float(d['run_time']), completed=bool(d['completed']))
        else: # Legacy file holding a whole J scan, arrays indexed [J, time, site]
            for i, J in enumerate(d['J']):
                runs.setdefault(key, {})[round(float(J), 6)] = dict(times=d['times'], occupation=d['occupation'][i], magnetization=d['magnetization'][i],
                                                                    truncation_error=d['truncation_error'][i], run_time=float(d['run_time'][i]), completed=True)
    return runs, info

def save_pdf(figs, path):
    tmp_path = path + '.tmp'
    with PdfPages(tmp_path) as pdf:
        for fig in figs:
            pdf.savefig(fig); plt.close(fig)
    os.replace(tmp_path, path)
    print(f'Saved {path}')

#---Per-run plots---#

def plot_runs(Run_Dir, runs, info):
    L = info['L']
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, L))
    for (max_bond, Timestep, tebd_order), by_J in sorted(runs.items()):
        K = round(0.5/Timestep)
        fig, axes = plt.subplots(len(by_J), 2, figsize=(11, 3.5*len(by_J)), squeeze=False, layout="constrained")
        for row, J in enumerate(sorted(by_J)):
            r = by_J[J]
            ax_occ, ax_mag = axes[row]
            for j in range(L):
                ax_occ.plot(r['times'], r['occupation'][:, j], color=colors[j], lw=1.8, label=f"site {j}")
                ax_mag.plot(r['times'], r['magnetization'][:, j], color=colors[j], lw=1.8)
            status = f"run time {r['run_time']:.1f}s" if r['completed'] else f"incomplete, t ≤ {r['times'][-1]:g}, {r['run_time']:.1f}s so far"
            ax_occ.set_xlabel("t"); ax_occ.set_ylabel("<n_j>"); ax_occ.set_title(f"J={J}, Boson occupation ({status})", fontsize=10)
            ax_mag.set_xlabel("t"); ax_mag.set_ylabel("<sz_j>"); ax_mag.set_title("Spin magnetization")
        axes[0, 0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"L{L}_N{info['N']}_T{info['Total_time']}_Bond{max_bond}_Step{Timestep}_K{K}_TEBD{tebd_order}_{'PBC' if info['cyclic'] else 'OBC'}")
        save_pdf([fig], os.path.join(Run_Dir, f'Bond{max_bond}_Step{Timestep}_K{K}_TEBD{tebd_order}.pdf'))

#---Convergence plots---#

def plot_convergence(Run_Dir, runs, info):
    if len(runs) < 2:
        return
    Run = os.path.basename(os.path.normpath(Run_Dir))
    if Run == 'Corrected':
        Run = os.path.basename(os.path.dirname(os.path.normpath(Run_Dir))) # Name the plot after the run, not the Corrected/ subfolder
    SITE = info['L'] // 2 # Middle site
    multiple_orders = len({order for (_, _, order) in runs}) > 1

    figs = []
    for J in sorted({J for r in runs.values() for J in r}):
        fig, (ax_n, ax_s, ax_t) = plt.subplots(3, 1, figsize=(10, 9), sharex=True, layout='constrained')
        # Trust limit from the most accurate run available for this J (largest max_bond, then smallest timestep, then highest order)
        best = max((k for k in runs if J in runs[k]), key=lambda k: (k[0], -k[1], k[2]))
        times, trunc = runs[best][J]['times'], runs[best][J]['truncation_error']
        over = np.where(trunc > TRUNC_LIMIT)[0]
        t_star = times[over[0]] if len(over) else None
        t_end = max(runs[k][J]['times'][-1] for k in runs if J in runs[k])
        for (chi, dt, order), r in sorted(runs.items()):
            if J not in r: continue
            r = r[J]
            label = f'χ={chi}, dt={dt}' + (f', TEBD{order}' if multiple_orders else '') + ('' if r['completed'] else ' (incomplete)')
            kw = dict(color=CHI_COLORS.get(chi, '0.3'), linestyle=DT_STYLES.get(dt, '-'), lw=1.5 if order == 2 else 2.5, alpha=1 if order == 2 else 0.6, label=label)
            ax_n.plot(r['times'], r['occupation'][:, SITE], **kw)
            ax_s.plot(r['times'], r['magnetization'][:, SITE], **kw)
            ax_t.semilogy(r['times'], np.maximum(r['truncation_error'], 1e-8), **kw)
        for ax in (ax_n, ax_s, ax_t):
            if t_star is not None:
                ax.axvspan(t_star, t_end, color='0.5', alpha=0.12, lw=0)
            ax.grid(alpha=0.25, lw=0.5)
            for side in ('top', 'right'): ax.spines[side].set_visible(False)
        ax_t.axhline(TRUNC_LIMIT, color='0.4', lw=0.8, linestyle=':')
        ax_n.set_ylabel(f'<n_{SITE}>'); ax_s.set_ylabel(f'<sz_{SITE}>'); ax_t.set_ylabel('1 - <psi|psi>'); ax_t.set_xlabel('t')
        handles, labels = ax_n.get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', fontsize=8, ncol=4, frameon=False)
        shade = f'shaded: t > {t_star:g}, where χ={best[0]}, dt={best[1]} truncation > {TRUNC_LIMIT:g}' if t_star is not None else f'truncation < {TRUNC_LIMIT:g} for all t (χ={best[0]}, dt={best[1]})'
        fig.suptitle(f'{Run}  J={J}  (site {SITE})\n{shade}', fontsize=10)
        figs.append(fig)
    save_pdf(figs, os.path.join(Run_Dir, f'Convergence_{Run}.pdf'))

#---Every folder containing .npz files---#

if __name__ == '__main__':
    Run_Dirs = sorted({os.path.dirname(f) for f in glob.glob(os.path.join(args.path, '**', '*.npz'), recursive=True)})
    if not Run_Dirs:
        print(f'No .npz files found under {args.path}')
    for Run_Dir in Run_Dirs:
        runs, info = load_folder(Run_Dir)
        if not args.no_runs:
            plot_runs(Run_Dir, runs, info)
        if not args.no_convergence:
            plot_convergence(Run_Dir, runs, info)
