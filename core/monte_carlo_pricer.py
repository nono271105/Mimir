# mimir/core/monte_carlo_pricer.py

import numpy as np
from numba import njit # Utiliser njit pour optimiser la performance

@njit
def run_monte_carlo(paths: np.ndarray, time_to_maturity: float, risk_free_rate: float, payoff_function, *payoff_args):
    """
    Exécute une simulation Monte Carlo pour calculer le prix d'une option.

    Paramètres:
    paths (np.ndarray): Un tableau (N_simulations, N_steps + 1) des chemins simulés
                        du prix du sous-jacent.
    time_to_maturity (float): Temps total jusqu'à l'échéance en années (T).
    risk_free_rate (float): Taux sans risque.
    payoff_function (callable): Une fonction qui prend un chemin unique
                                et des arguments de payoff (*payoff_args)
                                et retourne le gain de l'option pour ce chemin.
    *payoff_args: Arguments supplémentaires à passer à la payoff_function
                  (ex: K, barrier_level, option_type, etc.).

    Retourne:
    float: Le prix estimé de l'option.
    """
    
    # Nombre de simulations
    N_simulations = paths.shape[0]
    
    # Tableau pour stocker les payoffs de chaque simulation
    payoffs = np.zeros(N_simulations)
    
    # Calcul des payoffs pour chaque chemin
    # Numba ne peut pas compiler directement l'appel à une fonction Python arbitraire
    # passée comme argument si elle n'est pas déjà njit-ée ou si elle est complexe.
    # Pour le moment, nous allons laisser la boucle ici, et si payoff_function est elle-même njit-ée,
    # cela peut être performant.
    # Une alternative serait d'avoir des fonctions de payoff prédéfinies et njit-ées
    # que run_monte_carlo pourrait appeler spécifiquement.
    
    for i in range(N_simulations):
        # Applique la fonction de payoff au chemin actuel
        # Le chemin passé à la fonction de payoff doit être le chemin du sous-jacent uniquement
        payoffs[i] = payoff_function(paths[i, :], *payoff_args)
            
    # Calcul de la moyenne des payoffs
    average_payoff = np.mean(payoffs)
    
    # Actualisation de la moyenne des payoffs au temps t=0
    option_price = average_payoff * np.exp(-risk_free_rate * time_to_maturity)
    
    return option_price