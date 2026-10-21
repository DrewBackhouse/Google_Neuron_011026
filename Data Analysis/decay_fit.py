# Fits the decay of the boson occupation and spin magnetization oscillation amplitudes, then the decay rate as a function of J
# Run with: python "Data Analysis/decay_fit.py" ["Data Analysis/L6_N8_T100_Bond128_Step0.1_K5_TEBD2_PBC"]
# For the folder of single-J .npz files it writes, next to the data:
#   DecayFit_{Run}.pdf - page 1: site-averaged traces with fitted envelopes per J, page 2: decay rate against J with the Γ(J) fits
#   DecayFit_{Run}.csv - envelope fit parameters per (signal, J)
# Amplitude: half the max - min over a sliding window of one oscillation period, so the flat spin plateaus don't matter
# Envelope:  A(t) = A_inf + (A0 - A_inf) exp(-Γ t), fitted only where the truncation error is below TRUNC_LIMIT
# Rate:      Γ(J) = Γ0 + α J^p, and with p fixed at 2 (golden rule)

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

Run_Dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'L6_N8_T100_Bond128_Step0.1_K5_TEBD2_PBC')
TRUNC_LIMIT = 1e-2 # Same trust limit as plots.py
MIN_WINDOWS = 4    # Fewer trusted windows than this and A_inf is fixed at 0
SIGNALS = {'boson': ('occupation', '<n>', '#2a78d6'), 'spin': ('magnetization', '<sz>', '#eb6834')} # Categorical slots 1-2 of plots.py
FREE_INF = {'boson': False, 'spin': True} # A free boson A_inf trades off against Γ (J=0 levels off at A_inf~1 while J>0 fit A_inf=0), so it is fixed at 0 for a consistent rate across J.
                                          # The spin amplitude visibly plateaus (~0.1-0.25, finite L), so A_inf stays free

#---Loading---#

def load(Run_Dir):
    """{J: dict(times, boson, spin, t_trust)}, site averaged. All sites are equivalent with PBC and a uniform initial state"""
    runs = {}
    for f in sorted(glob.glob(os.path.join(Run_Dir, '*.npz'))):
        d = np.load(f, allow_pickle=False)
        n = int(d['n_done'])
        times, trunc = d['times'][:n], d['truncation_error'][:n]
        over = np.where(trunc > TRUNC_LIMIT)[0]
        runs[round(float(d['J']), 6)] = dict(times=times, t_trust=times[over[0]] if len(over) else times[-1],
                                             **{s: d[k][:n].mean(axis=1) for s, (k, _, _) in SIGNALS.items()})
    return runs

#---Envelope---#

def period(t, y):
    """Dominant period from the FFT peak of the mean-subtracted trace"""
    f = np.fft.rfftfreq(len(y), t[1] - t[0])
    P = np.abs(np.fft.rfft(y - y.mean()))
    return 1 / f[1:][np.argmax(P[1:])]

def envelope(t, y, T, t_max):
    """Half peak-to-peak over one-period windows stepped by half a period, keeping windows that end before t_max"""
    w = int(round(T / (t[1] - t[0])))
    tc, A, mid = [], [], []
    for i in range(0, len(t) - w, max(w // 2, 1)):
        if t[i + w] > t_max: break
        seg = y[i:i + w + 1]
        tc.append(t[i:i + w + 1].mean()); A.append((seg.max() - seg.min()) / 2); mid.append((seg.max() + seg.min()) / 2) # The boson centre line drifts up, so keep it for plotting
    return np.array(tc), np.array(A), np.array(mid)

def decay(t, A0, A_inf, G):
    return A_inf + (A0 - A_inf) * np.exp(-G * t)

def fit_envelope(tc, A, free_inf=True):
    if free_inf and len(tc) >= MIN_WINDOWS:
        p, C = curve_fit(decay, tc, A, p0=(A[0], A[-1] / 2, 0.02), bounds=([0, 0, 0], [np.inf, np.inf, np.inf]), maxfev=20000)
        return dict(A0=p[0], A_inf=p[1], G=p[2], sG=np.sqrt(C[2, 2]), free_inf=True)
    f = lambda t, A0, G: decay(t, A0, 0.0, G)
    p, C = curve_fit(f, tc, A, p0=(A[0], 0.1), bounds=([0, 0], [np.inf, np.inf]), maxfev=20000)
    sG = np.sqrt(C[1, 1]) if len(tc) > 2 else np.nan # Two points fix the curve exactly, no error estimate
    return dict(A0=p[0], A_inf=0.0, G=p[1], sG=sG, free_inf=False)

#---Γ(J)---#

def fit_rate(J, G, sG):
    sigma = np.where(np.isfinite(sG) & (sG > 0), sG, np.nanmax(sG[np.isfinite(sG)]) if np.isfinite(sG).any() else 1.0) # Points without an error get the largest one
    out = {}
    try:
        p, C = curve_fit(lambda J, G0, a, q: G0 + a * J**q, J, G, p0=(G[0], G[-1] / J[-1]**2, 2), sigma=sigma, absolute_sigma=False, bounds=([0, 0, 0.1], [np.inf, np.inf, 8]), maxfev=20000)
        out['free'] = (p, np.sqrt(np.diag(C)))
    except RuntimeError:
        out['free'] = None
    p, C = curve_fit(lambda J, G0, a: G0 + a * J**2, J, G, p0=(G[0], 1.0), sigma=sigma, absolute_sigma=False, maxfev=20000)
    out['p2'] = (p, np.sqrt(np.diag(C)))
    return out

#---Main---#

if __name__ == '__main__':
    Run = os.path.basename(os.path.normpath(Run_Dir))
    runs = load(Run_Dir)
    Js = sorted(runs)

    t_common = min(r['t_trust'] for r in runs.values()) # Shared trusted window, for a like-for-like early-time rate at every J
    fits = {s: {} for s in SIGNALS}
    early = {s: {} for s in SIGNALS}
    for J in Js:
        r = runs[J]
        for s in SIGNALS:
            T = period(r['times'], r[s])
            tc, A, mid = envelope(r['times'], r[s], T, r['t_trust'])
            fits[s][J] = dict(T=T, tc=tc, A=A, mid=mid, n=len(tc), **fit_envelope(tc, A, FREE_INF[s]))
            tc, A, _ = envelope(r['times'], r[s], T, t_common)
            early[s][J] = fit_envelope(tc, A, free_inf=False) # Pure exponential, A_inf = 0

    # Table
    rows = ['signal,J,period,Gamma,Gamma_err,A0,A_inf,A_inf_free,n_windows,t_trust,Gamma_early,Gamma_early_err,t_early']
    print(f'{Run}\n{"signal":7}{"J":>5}{"period":>8}{"Γ":>10}{"±":>9}{"A0":>8}{"A_inf":>8}{"n_win":>6}{"t_trust":>9}{"Γ_early":>10}{"±":>9}   (early: t ≤ {t_common:g}, A_inf = 0)')
    for s in SIGNALS:
        for J in Js:
            f, e = fits[s][J], early[s][J]
            print(f'{s:7}{J:5.1f}{f["T"]:8.2f}{f["G"]:10.4f}{f["sG"]:9.4f}{f["A0"]:8.3f}{f["A_inf"]:8.3f}{f["n"]:6d}{runs[J]["t_trust"]:9.1f}{e["G"]:10.4f}{e["sG"]:9.4f}')
            rows.append(f'{s},{J},{f["T"]:.4f},{f["G"]:.6g},{f["sG"]:.6g},{f["A0"]:.6g},{f["A_inf"]:.6g},{f["free_inf"]},{f["n"]},{runs[J]["t_trust"]:g},{e["G"]:.6g},{e["sG"]:.6g},{t_common:g}')
    with open(os.path.join(Run_Dir, f'DecayFit_{Run}.csv'), 'w') as fh:
        fh.write('\n'.join(rows) + '\n')

    rate_fits = {}
    for s in SIGNALS:
        for tag, src in (('full', fits), ('early', early)):
            J = np.array(Js); G = np.array([src[s][j]['G'] for j in Js]); sG = np.array([src[s][j]['sG'] for j in Js])
            rate_fits[s, tag] = fit_rate(J, G, sG)
            fr, f2 = rate_fits[s, tag]['free'], rate_fits[s, tag]['p2']
            if fr is not None:
                (G0, a, q), (eG0, ea, eq) = fr
                print(f'{s:5} {tag:5}: Γ(J) = {G0:.4f}(±{eG0:.4f}) + {a:.4f}(±{ea:.4f}) J^{q:.2f}(±{eq:.2f})')
            (G0, a), (eG0, ea) = f2
            print(f'{s:5} {tag:5}: Γ(J) = {G0:.4f}(±{eG0:.4f}) + {a:.4f}(±{ea:.4f}) J^2   [p fixed]')

    # Page 1: traces with envelopes
    fig1, axes = plt.subplots(len(Js), 2, figsize=(11, 3.2 * len(Js)), squeeze=False, layout='constrained')
    for row, J in enumerate(Js):
        r = runs[J]
        for col, s in enumerate(SIGNALS):
            ax, (_, ylab, c) = axes[row, col], SIGNALS[s]
            f = fits[s][J]
            tt = np.linspace(f['tc'][0], f['tc'][-1], 300) # Envelope only over the windows it was fitted to
            mid = np.interp(tt, f['tc'], f['mid'])     # Drawn about the windowed centre line
            env = decay(tt, f['A0'], f['A_inf'], f['G'])
            ax.plot(r['times'], r[s], color=c, lw=1.2)
            ax.plot(f['tc'], f['mid'] + f['A'], 'o', ms=3.5, color='0.25', mfc='none', mew=1)
            ax.plot(f['tc'], f['mid'] - f['A'], 'o', ms=3.5, color='0.25', mfc='none', mew=1)
            ax.plot(tt, mid + env, color='0.15', lw=1.5, linestyle=(0, (4, 2)))
            ax.plot(tt, mid - env, color='0.15', lw=1.5, linestyle=(0, (4, 2)))
            if r['t_trust'] < r['times'][-1]:
                ax.axvspan(r['t_trust'], r['times'][-1], color='0.5', alpha=0.12, lw=0)
            err = '' if not np.isfinite(f['sG']) else f' ± {f["sG"]:.3f}'
            ax.set_title(f'J={J}, {s}: Γ = {f["G"]:.4f}{err}' + ('' if f['free_inf'] else '  (A∞=0)'), fontsize=10)
            ax.set_xlabel('t'); ax.set_ylabel(ylab)
            ax.grid(alpha=0.25, lw=0.5)
            for side in ('top', 'right'): ax.spines[side].set_visible(False)
    fig1.suptitle(f'{Run}\nsite averaged; circles: window centre ± half-amplitude, dashed: centre ± [A∞ + (A0 − A∞)e^(−Γt)] (boson A∞ = 0); shaded: truncation > {TRUNC_LIMIT:g}', fontsize=10)

    # Page 2: Γ(J), full trusted window (top) and common early window (bottom)
    fig2, axes2 = plt.subplots(2, 2, figsize=(11, 8.5), layout='constrained')
    Jf = np.linspace(0, max(Js), 200)
    for row, (tag, src, desc) in enumerate((('full', fits, 'each run\'s trusted window'), ('early', early, f'common window t ≤ {t_common:g}, A∞ = 0'))):
        for ax, s in zip(axes2[row], SIGNALS):
            c = SIGNALS[s][2]
            G = np.array([src[s][j]['G'] for j in Js]); sG = np.array([src[s][j]['sG'] for j in Js])
            ax.errorbar(Js, G, yerr=np.nan_to_num(sG), fmt='o', ms=8, color=c, mec='white', mew=2, capsize=3, lw=1.5, label='envelope fit', zorder=3)
            fr, f2 = rate_fits[s, tag]['free'], rate_fits[s, tag]['p2']
            if fr is not None:
                (G0, a, q), (_, _, eq) = fr
                ax.plot(Jf, G0 + a * Jf**q, color='0.15', lw=1.5, label=f'$\\Gamma_0 + \\alpha J^p$, p = {q:.2f} ± {eq:.2f}')
            (G0, a), _ = f2
            ax.plot(Jf, G0 + a * Jf**2, color='0.45', lw=1.5, linestyle=(0, (4, 2)), label='$\\Gamma_0 + \\alpha J^2$ (p fixed)')
            ax.set_xlabel('J'); ax.set_ylabel('Γ'); ax.set_title(f'{s} amplitude decay rate ({desc})', fontsize=10)
            ax.grid(alpha=0.25, lw=0.5); ax.legend(fontsize=8, frameon=False)
            for side in ('top', 'right'): ax.spines[side].set_visible(False)
    fig2.suptitle(Run, fontsize=10)

    out = os.path.join(Run_Dir, f'DecayFit_{Run}.pdf')
    with PdfPages(out) as pdf:
        for fig in (fig1, fig2):
            pdf.savefig(fig); plt.close(fig)
    print(f'Saved {out}')
