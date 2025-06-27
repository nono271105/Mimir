import numpy as np


def binomial_option_pricing(
    option_type: str,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int,
    exercise_type: str = "EU",
    dividend_yield: float = 0.0,
    dividend_amount: float = 0.0,
    dividend_time: float = -1.0,
):
    """
    Calcule le prix d'une option européenne ou américaine en utilisant le modèle binomial.
    Gère les dividendes discrets pour les options américaines.

    Paramètres:
    option_type (str): 'C' pour Call, 'P' pour Put.
    S (float): Prix spot actuel du sous-jacent.
    K (float): Prix d'exercice de l'option.
    T (float): Temps jusqu'à l'échéance en années.
    r (float): Taux d'intérêt sans risque annuel.
    sigma (float): Volatilité annuelle du sous-jacent.
    N (int): Nombre de pas dans l'arbre binomial.
    exercise_type (str): 'EU' pour Européenne, 'US' pour Américaine.
    dividend_yield (float): Rendement continu des dividendes (pour compatibilité, non utilisé pour dividendes discrets).
    dividend_amount (float): Montant du dividende discret (en devise, ex: $2).
    dividend_time (float): Temps jusqu'au dividende (en années, doit être > 0 et < T).

    Retourne:
    float: Le prix de l'option.
    """
    if T <= 0 or sigma <= 0 or N <= 0:
        if T == 0:
            if option_type == "C":
                return max(0, S - K)
            else:  # Put
                return max(0, K - S)
        else:
            raise ValueError("T, sigma et N doivent être positifs.")

    if S <= 0 or K <= 0:
        raise ValueError("S et K doivent être positifs.")

    dt = T / N  # Durée de chaque pas de temps
    u = np.exp(sigma * np.sqrt(dt))  # Facteur de hausse
    d = 1 / u  # Facteur de baisse

    # Calculer le taux d'intérêt sans risque effectif par pas
    # Avec dividendes discrets, nous n'utilisons PAS le dividend_yield pour ajuster r
    # r_effective_per_step = np.exp(r * dt)
    # Le probabilité neutre au risque doit prendre en compte r
    p = (np.exp(r * dt) - d) / (u - d)  # Probabilité de hausse
    q = 1 - p  # Probabilité de baisse

    # Initialisation de l'arbre des prix du sous-jacent
    # S_tree[i][j] = prix du sous-jacent au pas i, chemin j (0 à i)
    S_tree = np.zeros((N + 1, N + 1))

    # Initialisation de l'arbre des valeurs d'option
    # option_values[i][j] = valeur de l'option au pas i, chemin j (0 à i)
    option_values = np.zeros((N + 1, N + 1))

    # --- Étape 1 : Construction de l'arbre des prix du sous-jacent ---
    for i in range(N + 1):  # i représente le numéro de pas (de 0 à N)
        for j in range(
            i + 1
        ):  # j représente le nombre de mouvements "vers le haut" (de 0 à i)
            S_tree[i, j] = S * (u ** (j)) * (d ** (i - j))

    # --- Étape 2 : Ajustement de l'arbre pour les dividendes discrets ---
    # Seulement si un dividende discret est spécifié et pertinent (temps > 0 et < T)
    if dividend_amount > 0 and 0 < dividend_time < T:
        # Trouver le pas le plus proche du temps du dividende
        # Le dividende est payé entre le pas (div_step-1) et div_step
        # Nous réduisons les prix à partir du pas div_step
        div_step = int(dividend_time / dt)

        # Ajuster les prix du sous-jacent à partir du pas du dividende
        for i in range(
            div_step, N + 1
        ):  # À partir du pas du dividende jusqu'à l'échéance
            for j in range(i + 1):
                # Assurez-vous que le prix ne devient pas négatif
                S_tree[i, j] = max(0, S_tree[i, j] - dividend_amount)

    # --- Étape 3 : Calcul des valeurs d'option à l'échéance (dernier pas N) ---
    for j in range(N + 1):
        if option_type == "C":
            option_values[N, j] = max(0, S_tree[N, j] - K)
        elif option_type == "P":
            option_values[N, j] = max(0, K - S_tree[N, j])
        else:
            raise ValueError(
                "Type d'option invalide. Utilisez 'C' pour Call ou 'P' pour Put."
            )

    # --- Étape 4 : Remontée de l'arbre pour calculer les valeurs d'option aux pas précédents ---
    for i in range(N - 1, -1, -1):  # De N-1 jusqu'à 0
        for j in range(i + 1):  # Pour chaque nœud à ce pas
            # Valeur de continuation (valeur si l'option n'est pas exercée à ce pas)
            # Actualisation des valeurs des noeuds futurs
            continuation_value = (
                p * option_values[i + 1, j + 1] + q * option_values[i + 1, j]
            ) * np.exp(-r * dt)

            # Valeur d'exercice immédiat
            if option_type == "C":
                intrinsic_value = max(0, S_tree[i, j] - K)
            else:  # option_type == 'P'
                intrinsic_value = max(0, K - S_tree[i, j])

            # Décision d'exercice pour les options Américaines
            if exercise_type == "US":
                option_values[i, j] = max(continuation_value, intrinsic_value)
            else:  # Européenne (pas d'exercice anticipé)
                option_values[i, j] = continuation_value

    return option_values[0, 0]  # Le prix de l'option au temps t=0
