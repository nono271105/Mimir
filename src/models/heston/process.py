# mimir/models/heston/process.py

import numpy as np
from numba import njit, prange

@njit(parallel=True)
def generate_heston_paths(S0: float, V0: float, kappa: float, theta: float, xi: float, rho: float, T: float, r: float, N_steps: int, N_simulations: int):
    """
    Simule N_simulations trajectoires du prix du sous-jacent et de sa variance
    selon le modèle de Heston, en utilisant le schéma de Milstein pour S_t
    et un schéma d'Euler avec troncature pour V_t.

    Paramètres:
    S0 (float): Prix spot initial du sous-jacent.
    V0 (float): Variance initiale (v0) du processus de variance.
    kappa (float): Vitesse de retour à la moyenne du processus de variance.
    theta (float): Variance moyenne à long terme (long-run variance) du processus de variance.
    xi (float): Volatilité de la volatilité (vol-of-vol).
    rho (float): Corrélation entre les deux processus de Wiener.
    T (float): Temps total jusqu'à l'échéance en années.
    r (float): Taux sans risque.
    N_steps (int): Nombre de pas de temps pour la simulation.
    N_simulations (int): Nombre de chemins (simulations) à générer.

    Retourne:
    tuple: (paths_S, paths_V)
        paths_S (np.ndarray): Un tableau (N_simulations, N_steps + 1) des prix du sous-jacent.
        paths_V (np.ndarray): Un tableau (N_simulations, N_steps + 1) des variances.
    """
    dt = T / N_steps
    sqrt_dt = np.sqrt(dt)

    paths_S = np.zeros((N_simulations, N_steps + 1))
    paths_V = np.zeros((N_simulations, N_steps + 1))

    paths_S[:, 0] = S0
    paths_V[:, 0] = V0

    Z1 = np.random.normal(0.0, 1.0, size=(N_simulations, N_steps))
    Z2 = np.random.normal(0.0, 1.0, size=(N_simulations, N_steps))

    for i in prange(N_simulations):
        for j in range(N_steps):
            S_t = paths_S[i, j]
            V_t = paths_V[i, j]

            dW_V_step = Z1[i, j] * sqrt_dt
            dW_S_step = (rho * Z1[i, j] + np.sqrt(1.0 - rho**2) * Z2[i, j]) * sqrt_dt

            sqrt_Vt_safe = np.sqrt(np.maximum(0.0, V_t))

            dV = kappa * (theta - V_t) * dt + xi * sqrt_Vt_safe * dW_V_step
            V_next_raw = V_t + dV
            paths_V[i, j+1] = np.maximum(0.0, V_next_raw)

            S_next = S_t + S_t * (r * dt + sqrt_Vt_safe * dW_S_step + 0.5 * V_t * (dW_S_step**2 - dt) / dt)
            
            paths_S[i, j+1] = S_next
            
    return paths_S, paths_V