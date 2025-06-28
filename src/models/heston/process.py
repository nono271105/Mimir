# mimir/models/heston/process.py

import numpy as np
from numba import njit, prange

@njit(parallel=True)
def generate_heston_paths(S0: float, V0: float, kappa: float, theta: float, xi: float,
                          rho: float, T: float, r: float, N_steps: int, N_simulations: int):
    """
    Génère des chemins de prix de sous-jacent (S) et de variance (V)
    selon le modèle de Heston, optimisé avec Numba.
    Utilise le schéma d'Euler pour la variance et un schéma d'Euler pour le log-prix (plus stable) pour le sous-jacent.

    Paramètres:
        S0 (float): Prix initial de l'actif sous-jacent.
        V0 (float): Variance initiale.
        kappa (float): Vitesse de retour à la moyenne de la variance.
        theta (float): Variance moyenne à long terme.
        xi (float): Volatilité de la volatilité (vol-of-vol).
        rho (float): Corrélation entre les mouvements du prix de l'actif et de sa variance.
        T (float): Temps total jusqu'à l'échéance (en années).
        r (float): Taux sans risque.
        N_steps (int): Nombre de pas de temps.
        N_simulations (int): Nombre de chemins de Monte Carlo à simuler.

    Returns:
        tuple: Deux tableaux NumPy (paths_S, paths_V) contenant les chemins simulés.
               paths_S: (N_simulations, N_steps + 1)
               paths_V: (N_simulations, N_steps + 1)
    """
    dt = T / N_steps
    sqrt_dt = np.sqrt(dt)

    paths_S = np.zeros((N_simulations, N_steps + 1), dtype=np.float64)
    paths_V = np.zeros((N_simulations, N_steps + 1), dtype=np.float64)

    paths_S[:, 0] = S0
    paths_V[:, 0] = V0

    Z1 = np.random.normal(0.0, 1.0, size=(N_simulations, N_steps))
    Z2 = np.random.normal(0.0, 1.0, size=(N_simulations, N_steps))

    for i in prange(N_simulations):
        for j in range(N_steps):
            V_t = paths_V[i, j]
            S_t = paths_S[i, j]

            sqrt_Vt_safe = np.sqrt(np.maximum(0.0, V_t))

            # Générer les mouvements browniens corrélés
            dW_V_step = Z1[i, j] * sqrt_dt
            dW_S_step = (rho * Z1[i, j] + np.sqrt(1.0 - rho**2) * Z2[i, j]) * sqrt_dt

            # Schéma d'Euler pour la variance (V_t)
            dV = kappa * (theta - V_t) * dt + xi * sqrt_Vt_safe * dW_V_step
            V_next_raw = V_t + dV
            paths_V[i, j+1] = np.maximum(0.0, V_next_raw) # Troncature pour assurer V >= 0

            # Schéma d'Euler pour le log-prix du sous-jacent (Log-Euler)
            S_next = S_t * np.exp((r - 0.5 * V_t) * dt + sqrt_Vt_safe * dW_S_step)
            paths_S[i, j+1] = S_next

    return paths_S, paths_V