# mimir/core/monte_carlo_pricer.py
import numpy as np

def run_monte_carlo(paths: np.ndarray, risk_free_rate: float, T: float, payoff_function, *payoff_args):
    """
    Orchestre la simulation Monte Carlo pour calculer le prix d'une option.

    Paramètres:
        paths (np.ndarray): Chemins simulés du sous-jacent (par exemple, paths_S de Heston).
                            Dimensions: (N_simulations, N_steps + 1)
        risk_free_rate (float): Taux sans risque.
        T (float): Temps jusqu'à l'échéance (en années).
        payoff_function (callable): Une fonction qui calcule le payoff d'une option pour un seul chemin.
                                    Elle doit prendre un np.ndarray pour le chemin et d'autres *payoff_args.
        *payoff_args: Arguments supplémentaires à passer à la fonction de payoff.

    Returns:
        float: Le prix de l'option calculé par Monte Carlo.
    """
    N_simulations = paths.shape[0]
    payoffs = np.zeros(N_simulations, dtype=np.float64)

    for i in range(N_simulations):
        payoffs[i] = payoff_function(paths[i], *payoff_args)

    # Calculer la moyenne des payoffs et l'actualiser au temps t=0
    option_price = np.mean(payoffs) * np.exp(-risk_free_rate * T)
    return option_price