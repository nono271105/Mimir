import numpy as np

def binomial_option_pricing(option_type: str, S: float, K: float, T: float, r: float, sigma: float, N: int, exercise_type: str = 'EU'):
    """
    Calcule le prix d'une option (Européenne ou Américaine) en utilisant
    le modèle binomial de Cox-Ross-Rubinstein (CRR).

    Args:
        option_type (str): 'C' pour Call, 'P' pour Put.
        S (float): Prix spot actuel du sous-jacent.
        K (float): Prix d'exercice de l'option.
        T (float): Temps jusqu'à l'échéance en années.
        r (float): Taux d'intérêt sans risque annuel.
        sigma (float): Volatilité annuelle du sous-jacent.
        N (int): Nombre de pas dans l'arbre binomial.
        exercise_type (str): 'EU' pour Européenne, 'US' pour Américaine.

    Returns:
        float: Le prix calculé de l'option.
    """
    if N <= 0:
        raise ValueError("Le nombre de pas (N) doit être un entier positif.")
    if T <= 0: # Pour T=0, le prix est le payoff intrinsèque
        if option_type == 'C':
            return max(0, S - K)
        elif option_type == 'P':
            return max(0, K - S)
        return 0.0 # Cas non valide

    dt = T / N  # Longueur d'un pas de temps
    u = np.exp(sigma * np.sqrt(dt)) # Facteur de hausse
    d = 1 / u # Facteur de baisse

    # Probabilité risque-neutre
    q = (np.exp(r * dt) - d) / (u - d)
    if not (0 <= q <= 1):
        raise ValueError(f"Probabilité risque-neutre q ({q:.4f}) hors de la plage [0,1]. Vérifiez les paramètres T, r, sigma.")


    # Initialisation des prix du sous-jacent à l'échéance (dernière colonne de l'arbre)
    S_T = np.zeros(N + 1)
    for j in range(N + 1):
        S_T[j] = S * (u ** (N - j)) * (d ** j)

    # Calcul des payoffs de l'option à l'échéance
    option_values = np.zeros(N + 1)
    if option_type == 'C':
        option_values = np.maximum(0, S_T - K)
    elif option_type == 'P':
        option_values = np.maximum(0, K - S_T)
    else:
        raise ValueError("Type d'option invalide. Doit être 'C' ou 'P'.")

    # Remontée de l'arbre pour calculer le prix de l'option à chaque nœud
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            # Valeur si l'option est maintenue (valeur espérée actualisée)
            value_if_continued = np.exp(-r * dt) * (q * option_values[j] + (1 - q) * option_values[j + 1])

            # Valeur si l'option est exercée immédiatement
            value_if_exercised = 0.0
            current_S = S * (u ** (i - j)) * (d ** j) # Prix du sous-jacent à ce nœud (pas i, nœud j)
            if option_type == 'C':
                value_if_exercised = max(0, current_S - K)
            elif option_type == 'P':
                value_if_exercised = max(0, K - current_S)

            # Pour les options américaines, choisir le max(maintenue, exercée)
            if exercise_type == 'US':
                option_values[j] = max(value_if_continued, value_if_exercised)
            else: # Pour les options européennes, on garde juste la valeur continuée
                option_values[j] = value_if_continued

    return option_values[0] # Le prix de l'option au temps t=0