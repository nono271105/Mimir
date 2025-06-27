import numpy as np

def binomial_option_pricing(option_type: str, S: float, K: float, T: float, r: float, sigma: float, N: int):
    """
    Calcule le prix d'une option européenne (Call ou Put) en utilisant
    le modèle binomial de Cox-Ross-Rubinstein (CRR).

    Args:
        option_type (str): 'C' pour Call, 'P' pour Put.
        S (float): Prix spot actuel du sous-jacent.
        K (float): Prix d'exercice de l'option.
        T (float): Temps jusqu'à l'échéance en années.
        r (float): Taux d'intérêt sans risque annuel.
        sigma (float): Volatilité annuelle du sous-jacent.
        N (int): Nombre de pas dans l'arbre binomial.

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
    # S'assurer que q est entre 0 et 1 (peut arriver avec des paramètres extrêmes)
    q = (np.exp(r * dt) - d) / (u - d)
    if not (0 <= q <= 1):
         # Gérer les cas où q est en dehors de [0,1], ce qui indique des paramètres irréalistes
        # Pour l'instant, on peut lever une erreur ou retourner un prix invalide
        # Pour un usage robuste, on pourrait envisager une gestion plus sophistiquée.
        # Pour ce modèle, nous allons nous assurer que q est clampé ou lever une erreur si les inputs sont mauvais.
        raise ValueError(f"Probabilité risque-neutre q ({q:.4f}) hors de la plage [0,1]. Vérifiez les paramètres T, r, sigma.")


    # Initialisation des prix du sous-jacent à l'échéance (dernière colonne de l'arbre)
    # Il y a N+1 prix possibles pour le sous-jacent à l'échéance
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
    # On commence par l'avant-dernière colonne et on remonte jusqu'à la racine
    for i in range(N - 1, -1, -1):
        for j in range(i + 1): # Il y a i+1 nœuds à chaque pas i
            # Prix de l'option à ce nœud (valeur espérée actualisée)
            option_values[j] = np.exp(-r * dt) * (q * option_values[j] + (1 - q) * option_values[j + 1])
            # Note : Pour les options européennes, on ne compare pas avec l'exercice anticipé.
            # Cela viendra pour les options américaines.

    return option_values[0] # Le prix de l'option au temps t=0