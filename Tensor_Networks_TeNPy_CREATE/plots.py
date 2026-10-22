# Makes all plots from the .npz files produced by Simulation.py, saving the PDFs next to the data
# Run with: python Tensor_Networks_TeNPy_CREATE/plots.py                 (every folder under Output/)
#      or:  python Tensor_Networks_TeNPy_CREATE/plots.py Tensor_Networks_TeNPy_CREATE/Output/L2_N8_T100_PBC
# For each folder of .npz files it writes:
#   Bond{χ}_Step{dt}_K{K}_TEBD{o}[_CBE{e}[-{e0}]].pdf  - occupation and magnetization of every site, one row per J
#   Convergence_{Run}.pdf                               - all (max_bond, timestep, cbe) runs overlaid for the largest J, one page per site, with % difference from the lowest-resource run (only if there is more than one run to compare)
# Legacy .npz files holding a whole J scan (e.g. Output/L7_N8_T100/Corrected) are also read
# Add --merge-sites to merge sites that sholuld be idential due to the symmetry of the initial state.

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
parser.add_argument('--merge-sites', action='store_true', help='In the per-run plots, draw sites related by a symmetry of the lattice and initial spins as one line (e.g. S100000 PBC: 1=5, 2=4)')
args = parser.parse_args()

CHI_COLORS = {32: '#2a78d6', 64: '#eb6834', 128: '#1baf7a', 256: '#eda100'}   # Categorical slots 1-4, colour follows bond dimension
DT_STYLES = {0.1: (0, (2, 2)), 0.05: '-', 0.025: (0, (6, 2))}                  # Line style follows timestep
ENV_WINDOW = 10 # The reference's initial peak-to-trough amplitude is measured over t <= this, spans at least one oscillation

def site_style(j, L):
    """Site 0 solid, later sites progressively shorter dashes, so sites lying on top of each other stay visible"""
    if j == 0:
        ls = '-'
    else:
        d = np.geomspace(10, 1.5, max(L - 1, 1))[j - 1]
        ls = (0, (d, 0.6*d))
    return dict(color=plt.cm.tab10(j % 10), lw=1.8, linestyle=ls)

def site_groups(L, init, cyclic):
    """Sites grouped into classes that are equivalent under the symmetries of the chain (rotations and reflections with PBC, the reflection with OBC)
    that leave the initial spins unchanged. The Hamiltonian is the same on every site and bond, so sites in a class have identical dynamics"""
    init = (init or '').ljust(L, '0') # Runs from before init was configurable started all up
    maps = [lambda j, s=s, k=k: (s*j + k) % L for s in (1, -1) for k in range(L)] if cyclic else [lambda j: j, lambda j: L - 1 - j]
    maps = [g for g in maps if all(init[g(j)] == init[j] for j in range(L))]
    groups = {tuple(sorted({g(j) for g in maps})) for j in range(L)}
    return sorted(groups)

#---Loading---#

def cbe_value(d, name):
    """QR-TEBD expansion rate stored in the .npz, None when not used (TDVP, quimb or left out). Stored as NaN as np.savez can't hold None"""
    return None if name not in d or np.isnan(d[name]) else float(d[name])

def cbe_tag(cbe, cbe0):
    return '' if cbe is None else f'_CBE{cbe}' + ('' if cbe0 is None else f'-{cbe0}') # Same format as CBE_Tag in Simulation.py

def load_folder(Run_Dir):
    """Returns (runs, info). runs[(max_bond, Timestep, tebd_order, cbe_expand, cbe_expand_0)][J] = dict of arrays cut to the completed samples. info holds the shared model parameters."""
    runs, info = {}, {}
    for f in sorted(glob.glob(os.path.join(Run_Dir, '*.npz'))):
        d = np.load(f, allow_pickle=False)
        key = (int(d['max_bond']), float(d['Timestep']), int(d['tebd_order']), cbe_value(d, 'cbe_expand'), cbe_value(d, 'cbe_expand_0'))
        info.update(L=int(d['L']), N=int(d['N']), Total_time=d['Total_time'].item(), Steps_per_sample=int(d['Steps_per_sample']), cyclic=bool(d['cyclic']) if 'cyclic' in d else False,
                    init=str(d['init']) if 'init' in d else None) # Initial spin state, missing in runs from before it was configurable
        if d['J'].ndim == 0: # One J per file, possibly still running / crashed
            n = int(d['n_done'])
            runs.setdefault(key, {})[round(float(d['J']), 6)] = dict(times=d['times'][:n], occupation=d['occupation'][:n], magnetization=d['magnetization'][:n],
                                                                     run_time=float(d['run_time']), completed=bool(d['completed']),
                                                                     peak_memory_MB=float(d['peak_memory_MB']) if 'peak_memory_MB' in d else None) # Missing in runs from before it was recorded
        else: # Legacy file holding a whole J scan, arrays indexed [J, time, site]
            for i, J in enumerate(d['J']):
                runs.setdefault(key, {})[round(float(J), 6)] = dict(times=d['times'], occupation=d['occupation'][i], magnetization=d['magnetization'][i],
                                                                    run_time=float(d['run_time'][i]), completed=True)
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
    groups = site_groups(L, info['init'], info['cyclic']) if args.merge_sites else [(j,) for j in range(L)]
    for (max_bond, Timestep, tebd_order, cbe, cbe0), by_J in sorted(runs.items(), key=lambda kv: str(kv[0])): # str as None can't be compared with floats
        K = round(0.5/Timestep)
        fig, axes = plt.subplots(len(by_J), 2, figsize=(11, 3.5*len(by_J)), squeeze=False, layout="constrained")
        for row, J in enumerate(sorted(by_J)):
            r = by_J[J]
            ax_occ, ax_mag = axes[row]
            for i, g in enumerate(groups):
                g = list(g)
                label = f"site {g[0]}" if len(g) == 1 else f"sites {','.join(map(str, g))}"
                ax_occ.plot(r['times'], r['occupation'][:, g].mean(axis=1), label=label, **site_style(i, len(groups))) # Mean over the class, equal up to numerical error
                ax_mag.plot(r['times'], r['magnetization'][:, g].mean(axis=1), **site_style(i, len(groups)))
            status = f"run time {r['run_time']:.1f}s" if r['completed'] else f"incomplete, t ≤ {r['times'][-1]:g}, {r['run_time']:.1f}s so far"
            if r.get('peak_memory_MB') is not None:
                status += f", {r['peak_memory_MB']:.0f} MB"
            ax_occ.set_xlabel("t"); ax_occ.set_ylabel("<n_j>"); ax_occ.set_title(f"J={J}, Boson occupation ({status})", fontsize=10)
            ax_mag.set_xlabel("t"); ax_mag.set_ylabel("<sz_j>"); ax_mag.set_title("Spin magnetization")
        axes[0, 0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"L{L}_N{info['N']}" + ('' if info['init'] is None else f"_S{info['init']}") + f"_T{info['Total_time']}_Bond{max_bond}_Step{Timestep}_K{K}_TEBD{tebd_order}{cbe_tag(cbe, cbe0)}_{'PBC' if info['cyclic'] else 'OBC'}")
        save_pdf([fig], os.path.join(Run_Dir, f'Bond{max_bond}_Step{Timestep}_K{K}_TEBD{tebd_order}{cbe_tag(cbe, cbe0)}.pdf'))

#---Convergence plots---#

def initial_span(x, times):
    """Peak-to-trough amplitude of x over the first ENV_WINDOW in t, falling back to the whole run (then 1) if that is flat"""
    return np.ptp(x[times <= ENV_WINDOW]) or np.ptp(x) or 1.0

def plot_convergence(Run_Dir, runs, info):
    J = max(J for r in runs.values() for J in r) # Largest J only
    keys = [k for k in runs if J in runs[k]]
    if len(keys) < 2:
        return
    Run = os.path.basename(os.path.normpath(Run_Dir))
    if Run == 'Corrected':
        Run = os.path.basename(os.path.dirname(os.path.normpath(Run_Dir))) # Name the plot after the run, not the Corrected/ subfolder
    multiple_orders = len({k[2] for k in keys}) > 1
    cbes = sorted({k[3:] for k in keys}, key=str)
    CBE_WIDTHS = {c: w for c, w in zip(cbes, np.linspace(1.2, 3, len(cbes)))} # Line width follows (cbe_expand, cbe_expand_0) when more than one is present
    ref = min(keys, key=lambda k: (k[0], -k[1], k[3] or 0, k[4] or 0, k[2])) # Lowest resource run: lowest bond, then largest timestep, then lowest expansion rate, then lowest order
    ref_r = runs[ref][J]

    figs = []
    for SITE in range(info['L']):
        fig, (ax_n, ax_s, ax_dn, ax_ds) = plt.subplots(4, 1, figsize=(10, 11), sharex=True, layout='constrained')
        spans = {q: initial_span(ref_r[q][:, SITE], ref_r['times']) for q in ('occupation', 'magnetization')}
        for k in sorted(keys, key=str):
            chi, dt, order, cbe, cbe0 = k
            r = runs[k][J]
            label = f'χ={chi}, dt={dt}' + (f', TEBD{order}' if multiple_orders else '') + (f', cbe={cbe}' + ('' if cbe0 is None else f'-{cbe0}') if len(cbes) > 1 else '') + ('' if r['completed'] else ' (incomplete)') + (' (ref)' if k == ref else '')
            lw = CBE_WIDTHS[(cbe, cbe0)] if len(cbes) > 1 else (1.5 if order == 2 else 2.5)
            kw = dict(color=CHI_COLORS.get(chi, '0.3'), linestyle=DT_STYLES.get(dt, '-'), lw=lw, alpha=1 if order == 2 else 0.6, label=label)
            ax_n.plot(r['times'], r['occupation'][:, SITE], **kw)
            ax_s.plot(r['times'], r['magnetization'][:, SITE], **kw)
            m = min(len(r['times']), len(ref_r['times'])) # Same sample spacing in every run, incomplete runs are just shorter
            for ax, q in ((ax_dn, 'occupation'), (ax_ds, 'magnetization')):
                ax.plot(r['times'][:m], 100 * (r[q][:m, SITE] - ref_r[q][:m, SITE]) / spans[q], **kw)
        for ax in (ax_n, ax_s, ax_dn, ax_ds):
            ax.grid(alpha=0.25, lw=0.5)
            for side in ('top', 'right'): ax.spines[side].set_visible(False)
        for ax in (ax_dn, ax_ds):
            ax.axhline(0, color='0.4', lw=0.8, linestyle=':')
        ax_n.set_ylabel(f'<n_{SITE}>'); ax_s.set_ylabel(f'<sz_{SITE}>')
        ax_dn.set_ylabel(f'% diff <n_{SITE}> vs ref'); ax_ds.set_ylabel(f'% diff <sz_{SITE}> vs ref'); ax_ds.set_xlabel('t')
        handles, labels = ax_n.get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', fontsize=8, ncol=4, frameon=False)
        fig.suptitle(f'{Run}  J={J}  site {SITE}\n% diff = 100·(run − ref) / initial peak-to-trough amplitude of ref (t ≤ {ENV_WINDOW:g})', fontsize=10)
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
