import numpy as np
import tenpy.linalg.np_conserved as npc
from tenpy.networks.site import Site
from tenpy.networks.mps import MPS
from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.algorithms.tebd import QRBasedTEBDEngine
from tenpy.algorithms.tdvp import TwoSiteTDVPEngine
import os
import sys
import time
import resource


def TNHamiltonian(N, D, Spin_Boson_Interaction_Coefficent):

    a = np.zeros((N, N), dtype=complex)
    for n in range(N - 1):
        a[n, n + 1] = np.sqrt(n + 1)
    n_op = np.diag(np.arange(N)).astype(complex)
    proj_top = np.zeros((N, N), dtype=complex)
    proj_top[N - 1, N - 1] = 1.0
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity_2 = np.eye(2, dtype=complex)
    identity_N = np.eye(N, dtype=complex)
    op = dict(sx=np.kron(sigma_x, identity_N), sy=np.kron(sigma_y, identity_N), sz=np.kron(sigma_z, identity_N), a=np.kron(identity_2, a), adag=np.kron(identity_2, a.conj().T), n=np.kron(identity_2, n_op), proj_top=np.kron(identity_2, proj_top))

    H1 = D * (op["a"] + op["adag"]) + Spin_Boson_Interaction_Coefficent * (op["proj_top"] @ op["sx"]) # Same on every site

    return H1, op


class SpinBosonChain(CouplingMPOModel): # Class for defult MPO model

    def init_sites(self, model_params): # Update the sites with our H1
        N, D, g = (model_params[k] for k in ("N", "D", "g"))
        H1, op = TNHamiltonian(N, D, g)
        leg = npc.LegCharge.from_trivial(2 * N) # No conserved charge, displacement term breaks U(1)
        ops = {name: op[name] for name in ("sx", "sy", "sz", "n")}
        ops["H1"] = H1
        return Site(leg, **ops)

    def init_lattice(self, model_params): # Update the lattice to our geometry
        site = self.init_sites(model_params)
        return Chain(model_params["L"], site, bc=("periodic" if model_params["cyclic"] else "open"), bc_MPS="finite")

    def init_terms(self, model_params): # Add the inter-site interactions
        J = model_params["J"]
        self.add_onsite(1.0, 0, "H1")
        self.add_coupling(J / 2, 0, "sx", 0, "sx", 1) # 0.5 J (sx sx + sy sy), with periodic bc the (L-1, 0) bond is added automatically
        self.add_coupling(J / 2, 0, "sy", 0, "sy", 1)


class SpinBosonChainNN(SpinBosonChain, NearestNeighborModel): # Bond Hamiltonians for TEBD, only valid when every bond is nearest neighbour in the MPS
    pass


def TNSim(L, N, D, spin_interaction_coefficient, Spin_Boson_Interaction_Coefficent, Total_time, max_bond, dt, tebd_order, steps_per_sample, cyclic=False, data_path=None, metadata=None, save_every=1, cbe_expand=0.1, cbe_expand_0=None, svd_min=1e-10, initial_state=None):
    t0 = time.time()
    n_samples = int(Total_time / (steps_per_sample * dt) + 1e-9) + 1
    times = np.arange(n_samples) * steps_per_sample * dt

    use_tdvp = cyclic and L > 2 # TEBD needs nearest neighbour bonds, so the wrap-around bond is handled with TDVP on the periodic MPO. L=2 already has its only bond
    model_params = dict(L=L, N=N, D=D, J=spin_interaction_coefficient, g=Spin_Boson_Interaction_Coefficent, cyclic=use_tdvp)
    model = SpinBosonChain(model_params) if use_tdvp else SpinBosonChainNN(model_params)
    psi = MPS.from_product_state(model.lat.mps_sites(), [int(s) * N for s in (initial_state or '0' * L)], bc="finite") # '0' = spin up (index 0), '1' = spin down (index N), boson vacuum

    trunc_params = dict(svd_min=svd_min, chi_max=(max_bond if max_bond is not None else (2 * N) ** (L // 2)))
    if use_tdvp:
        eng = TwoSiteTDVPEngine(psi, model, dict(dt=dt, N_steps=steps_per_sample, trunc_params=trunc_params))
        bond_norm = 0.0 # TDVP has no Trotter error
    else:
        tebd_params = dict(order=tebd_order, dt=dt, N_steps=steps_per_sample, trunc_params=trunc_params, cbe_expand=cbe_expand) # Bond grows by at most a factor (1 + cbe_expand) per step
        if cbe_expand_0 is not None:
            tebd_params["cbe_expand_0"] = cbe_expand_0 # Growth rate at bond dimension 1, falling linearly to cbe_expand at max_bond
        eng = QRBasedTEBDEngine(psi, model, tebd_params)
        bond_norm = np.mean([npc.norm(Hb) for Hb in model.H_bond if Hb is not None]) # Mean Frobenius norm of bond terms, as in quimb's LocalHam1D.mean_norm

    occupation = np.zeros((len(times), L))
    magnetization = np.zeros((len(times), L))
    truncation_error = np.zeros(len(times))
    trotter_error = np.zeros(len(times))
    for it in range(len(times)):
        if it > 0:
            eng.run() # steps_per_sample steps of dt
        occupation[it] = np.real(psi.expectation_value("n"))
        magnetization[it] = np.real(psi.expectation_value("sz"))
        truncation_error[it] = eng.trunc_err.eps # TeNPy renormalises after truncation, so the discarded weight is tracked by the engine
        trotter_error[it] = it * steps_per_sample * bond_norm * dt ** (tebd_order + 1) # Same Trotter error estimate as quimb's TEBD.err
        if data_path is not None and ((it + 1) % save_every == 0 or it == len(times) - 1):
            save_data(data_path, times, occupation, magnetization, truncation_error, trotter_error, it + 1, time.time() - t0, metadata or {})

    return times, occupation, magnetization, truncation_error, trotter_error, time.time() - t0

def peak_memory_mb(): # Peak resident memory of this process so far, ru_maxrss is in KB on Linux and bytes on macOS
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == 'darwin' else 1024)

def save_data(Data_Path, times, occupation, magnetization, truncation_error, trotter_error, n_done, run_time, metadata):
    tmp_path = Data_Path + '.tmp'
    with open(tmp_path, 'wb') as f: # Pass a file handle as np.savez would otherwise append .npz to the temp name
        np.savez(f, times=times, occupation=occupation, magnetization=magnetization, truncation_error=truncation_error, trotter_error=trotter_error,
                 n_done=n_done, completed=(n_done == len(times)), run_time=run_time, peak_memory_MB=peak_memory_mb(), **metadata) # Saved every sample, so runs killed by the time limit still record it
    os.replace(tmp_path, Data_Path) # Atomic, so a crash mid-save never corrupts the last good file
