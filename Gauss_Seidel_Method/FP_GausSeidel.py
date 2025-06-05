"""
Newton-Raphson Power Flow Analysis
Created on Wed May  25 2025

@author: Ronaldo César
"""

import numpy as np
import pandas as pd
import time

def gauss_seidel_pf(Ybus, Sbus, V0, ref, pv, pq, tol=1e-4, max_it=500):
    V = V0.copy()
    Vm = np.abs(V)
    converged = False
    i = 0

    npv = len(pv)
    npq = len(pq)

    while not converged and i < max_it:
        i += 1
        V_prev = V.copy()

        # Atualização PQ
        for k in pq:
            Vk = (Sbus[k].conj() / V[k].conj() - (Ybus[k, :] @ V) + Ybus[k, k] * V[k]) / Ybus[k, k]
            V[k] = Vk

        # Atualização PV
        for k in pv:
            Sk = Sbus[k]
            Qk = (V[k] * (Ybus[k, :] @ V).conj()).imag
            Sk = complex(Sk.real, Qk)

            Vk = (Sk.conjugate() / V[k].conjugate() - (Ybus[k, :] @ V) + Ybus[k, k] * V[k]) / Ybus[k, k]
            V[k] = Vk

            # Corrige magnitude
            V[k] = Vm[k] * V[k] / abs(V[k])

        # Verificar convergência
        mismatch = V * (Ybus @ V).conj() - Sbus
        #print(f"Mismatch {mismatch}")
        normF = max(np.abs(mismatch[pq].real).max(), np.abs(mismatch[pq].imag).max(),
                      np.abs(mismatch[pv].real).max() if pv else 0)
        print(f"Iter {i}: max mismatch = {normF:.6f}")

        if normF < tol:
            converged = True
            print(f"Convergiu em {i} iterações.")

    if not converged:
        print(f"Não convergiu em {max_it} iterações.")

    return V, converged, i


# =============================
# Dados do sistema exemplo
# =============================

Ybus = np.array([[ 1.27428062 -4.19457769j,  0.         +0.j        , 0.         +0.j        , -0.55826936 +2.58199581j, 0.         +0.j        , -0.71601126 +1.61258187j],
       [ 0.         +0.j        ,  1.0214013  -1.95452446j, -0.44486039 +0.6460628j ,  0.         +0.j        , -0.57654092 +1.30846166j,  0.         +0.j        ],
       [ 0.         +0.j        , -0.44486039 +0.6460628j , 0.44486039 -8.16485979j, -0.         +7.51879699j, 0.         +0.j        ,  0.         +0.j        ],
       [-0.55826936 +2.58199581j,  0.         +0.j        , -0.         +7.51879699j,  1.11237143-12.42573654j, 0.         +0.j        , -0.55410207 +2.32494373j],
       [ 0.         +0.j        , -0.57654092 +1.30846166j,  0.         +0.j        ,  0.         +0.j        , 0.57654092 -4.64179499j, -0.         +3.33333333j],
       [-0.71601126 +1.61258187j,  0.         +0.j        , 0.         +0.j        , -0.55410207 +2.32494373j, -0.         +3.33333333j,  1.27011333 -7.27085894j]])

Sbus = np.array([0+0j, 0.5+0.0j, -0.55-0.13j, 0.0+0.0j, -0.30-0.18j, -0.5-0.05j])

V0 = np.array([1.05+0j, 1.1+0j, 1+0j, 1+0j, 1+0j, 1+0j])

ref = [0]
pv = [1]
pq = [2, 3, 4, 5]

# =============================
# Execução do fluxo
# =============================
start_time = time.time()
V, converged, iterations = gauss_seidel_pf(Ybus, Sbus, V0, ref, pv, pq)
end_time = time.time()
processing_time = end_time - start_time
print(f"\nTempo de processamento até a convergência: {processing_time*1000:.9f} ms.")
if converged:
    print("\nTensões finais:")
    for idx, v in enumerate(V):
        print(f"Barra {idx+1}: {abs(v):.4f} ∠ {np.angle(v, deg=True):.2f}°")

    # =============================
    # Tabela das Barras
    # =============================
    S_inj = V * (Ybus @ V).conj()  # Potência injetada em cada barra

    bus_data = []

    for idx in range(len(V)):
        voltage_mag = abs(V[idx])
        voltage_ang = np.angle(V[idx], deg=True)
        Pg = S_inj[idx].real if idx in ref else 0
        Qg = S_inj[idx].imag if idx in ref else 0
        Pl = -Sbus[idx].real if idx in pq else 0
        Ql = -Sbus[idx].imag if idx in pq else 0

        bus_data.append([idx+1, voltage_mag, voltage_ang, Pg, Qg, Pl, Ql])

    df_bus = pd.DataFrame(bus_data, columns=['Bus', 'V(pu)', 'Angle(deg)', 'P_gen', 'Q_gen', 'P_load', 'Q_load'])

    print("\n============= Bus Data =============")
    print(df_bus.to_string(index=False))

    # =============================
    # Tabela dos Ramos
    # =============================
    branch_data = []
    n_bus = len(V)
    total_P_loss = 0
    total_Q_loss = 0

    for i in range(n_bus):
        for j in range(i+1, n_bus):
            if Ybus[i, j] != 0:
                Y = -Ybus[i, j]  # Admitância da linha

                S_ij = V[i] * ((V[i] - V[j]) * Y).conj()
                S_ji = V[j] * ((V[j] - V[i]) * Y).conj()

                P_loss = (S_ij + S_ji).real
                Q_loss = (S_ij + S_ji).imag
                
                total_P_loss += P_loss
                total_Q_loss += Q_loss

                branch_data.append([i+1, j+1, S_ij.real, S_ij.imag,
                                     S_ji.real, S_ji.imag, P_loss, Q_loss])

    df_branch = pd.DataFrame(branch_data, columns=[
        'From', 'To', 'P_from', 'Q_from', 'P_to', 'Q_to', 'P_loss', 'Q_loss'])

    print("\n============= Branch Data =============")
    print(df_branch.to_string(index=False))

    print(f"\nPerda Total do Sistema: P_total_loss = {total_P_loss:.4f} pu, Q_total_loss = {total_Q_loss:.4f} pu")

else:
    print("Fluxo não convergiu.")