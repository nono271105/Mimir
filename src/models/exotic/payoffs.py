import numpy as np

def calculate_barrier_payoff(path_S: np.ndarray, K: float, barrier_level: float, option_type: str, knock_type: str, in_out: str):
    """
    Calcule le payoff d'une option barrière pour un seul chemin simulé.

    Paramètres:
        path_S (np.ndarray): Chemin simulé du prix du sous-jacent.
        K (float): Prix d'exercice (strike).
        barrier_level (float): Niveau de la barrière.
        option_type (str): 'C' pour Call, 'P' pour Put.
        knock_type (str): 'out' pour Knock-Out, 'in' pour Knock-In.
        in_out (str): 'up' si la barrière est "Up", 'down' si "Down".

    Returns:
        float: Le payoff de l'option pour ce chemin.
    """
    has_hit_barrier = False
    if in_out == 'up':
        has_hit_barrier = np.any(path_S >= barrier_level)
    elif in_out == 'down':
        has_hit_barrier = np.any(path_S <= barrier_level)
    else:
        raise ValueError("in_out doit être 'up' ou 'down'.")

    payoff = 0.0
    if knock_type == 'out':
        if not has_hit_barrier:
            if option_type == 'C':
                payoff = max(0.0, path_S[-1] - K)
            elif option_type == 'P':
                payoff = max(0.0, K - path_S[-1])
            else:
                raise ValueError("option_type doit être 'C' ou 'P'.")
    elif knock_type == 'in':
        if has_hit_barrier:
            if option_type == 'C':
                payoff = max(0.0, path_S[-1] - K)
            elif option_type == 'P':
                payoff = max(0.0, K - path_S[-1])
            else:
                raise ValueError("option_type doit être 'C' ou 'P'.")
    else:
        raise ValueError("knock_type doit être 'out' ou 'in'.")

    return payoff

def calculate_asian_payoff(path_S: np.ndarray, K: float, option_type: str):
    """
    Calcule le payoff d'une option asiatique (moyenne arithmétique) pour un seul chemin simulé.

    Paramètres:
        path_S (np.ndarray): Chemin simulé du prix du sous-jacent.
        K (float): Prix d'exercice (strike).
        option_type (str): 'C' pour Call, 'P' pour Put.

    Returns:
        float: Le payoff de l'option pour ce chemin.
    """
    average_price = np.mean(path_S) # Moyenne arithmétique sur le chemin

    payoff = 0.0
    if option_type == 'C':
        payoff = max(0.0, average_price - K)
    elif option_type == 'P':
        payoff = max(0.0, K - average_price)
    else:
        raise ValueError("option_type doit être 'C' ou 'P'.")
    return payoff

def calculate_digital_payoff(path_S: np.ndarray, K: float, payoff_amount: float, option_type: str):
    """
    Calcule le payoff d'une option digitale (cash-or-nothing) pour un seul chemin simulé.

    Paramètres:
        path_S (np.ndarray): Chemin simulé du prix du sous-jacent (seule la dernière valeur est pertinente).
        K (float): Prix d'exercice (strike).
        payoff_amount (float): Montant fixe versé si l'option est dans la monnaie.
        option_type (str): 'C' pour Call, 'P' pour Put.

    Returns:
        float: Le payoff de l'option pour ce chemin.
    """
    final_price = path_S[-1] # Le prix final est ce qui compte pour une digitale

    payoff = 0.0
    if option_type == 'C':
        if final_price > K:
            payoff = payoff_amount
    elif option_type == 'P':
        if final_price < K:
            payoff = payoff_amount
    else:
        raise ValueError("option_type doit être 'C' ou 'P'.")
    return payoff