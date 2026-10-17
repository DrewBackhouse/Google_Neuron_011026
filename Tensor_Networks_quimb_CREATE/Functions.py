import numpy as np
import quimb.tensor as qtn
import os
import time


def TNHamiltonian(L, N, D, spin_interaction_coefficient, Spin_Boson_Interaction_Coefficent, cyclic=False):

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

    H1 = {}
    for j in range(L):
        H1[j] = D * (op["a"] + op["adag"]) + Spin_Boson_Interaction_Coefficent * (op["proj_top"] @ op["sx"])

    H2 = {}
    xy_bond = 0.5 * (np.kron(op["sx"], op["sx"]) + np.kron(op["sy"], op["sy"]))
    for j in range(L - 1):
        H2[(j, j + 1)] = spin_interaction_coefficient * xy_bond
    if cyclic and L > 2: # Wrap-around bond, keyed (L-1, 0) to match the gate quimb's TEBD looks up. L=2 already has its only bond
        H2[(L - 1, 0)] = spin_interaction_coefficient * xy_bond

    return H1, H2, op


def TNSim(L, N, D, spin_interaction_coefficient, Spin_Boson_Interaction_Coefficent, Total_time, max_bond, dt, tebd_order, steps_per_sample, cyclic=False, data_path=None, metadata=None, save_every=1):
    t0 = time.time()
    n_samples = int(Total_time / (steps_per_sample * dt) + 1e-9) + 1
    times = np.arange(n_samples) * steps_per_sample * dt

    d = 2 * N
    vec0 = np.zeros(d, dtype=complex)
    vec0[0] = 1.0
    psi0 = qtn.MPS_product_state([vec0.copy() for _ in range(L)], cyclic=cyclic) # TEBD requires the MPS and Hamiltonian boundary conditions to match

    H1, H2, op = TNHamiltonian(L, N, D, spin_interaction_coefficient, Spin_Boson_Interaction_Coefficent, cyclic)
    ham = qtn.LocalHam1D(L, H2=H2, H1=H1, cyclic=cyclic)
    tebd = qtn.TEBD(psi0, ham, dt=dt, split_opts=({} if max_bond is None else dict(max_bond=max_bond)))

    occupation = np.zeros((len(times), L))
    magnetization = np.zeros((len(times), L))
    truncation_error = np.zeros(len(times))
    trotter_error = np.zeros(len(times))
    for it in range(len(times)):
        if it > 0:
            for step in range(steps_per_sample):
                tebd.step(order=tebd_order, queue=(step < steps_per_sample - 1))
        psi = tebd.pt
        norm = np.real(psi.H @ psi)
        for j in range(L):
            occupation[it, j] = np.real(psi.local_expectation_exact(op["n"], (j,)))
            magnetization[it, j] = np.real(psi.local_expectation_exact(op["sz"], (j,)))
        truncation_error[it] = 1 - norm
        trotter_error[it] = tebd.err
        if data_path is not None and ((it + 1) % save_every == 0 or it == len(times) - 1):
            save_data(data_path, times, occupation, magnetization, truncation_error, trotter_error, it + 1, time.time() - t0, metadata or {})

    return times, occupation, magnetization, truncation_error, trotter_error, time.time() - t0

def save_data(Data_Path, times, occupation, magnetization, truncation_error, trotter_error, n_done, run_time, metadata):
    tmp_path = Data_Path + '.tmp'
    with open(tmp_path, 'wb') as f: # Pass a file handle as np.savez would otherwise append .npz to the temp name
        np.savez(f, times=times, occupation=occupation, magnetization=magnetization, truncation_error=truncation_error, trotter_error=trotter_error,
                 n_done=n_done, completed=(n_done == len(times)), run_time=run_time, **metadata)
    os.replace(tmp_path, Data_Path) # Atomic, so a crash mid-save never corrupts the last good file