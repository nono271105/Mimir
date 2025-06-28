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
    discrete_dividends: list = None,  # Changé pour une liste de (montant, temps)
):
    """
    Calcule le prix d'une option européenne ou américaine en utilisant le modèle binomial.
    Gère les dividendes discrets multiples pour les options américaines.

    Paramètres:
    option_type (str): 'C' pour Call, 'P' pour Put.
    S (float): Prix spot actuel du sous-jacent.
    K (float): Prix d'exercice de l'option.
    T (float): Temps jusqu'à l'échéance en années.
    r (float): Taux d'intérêt sans risque annuel.
    sigma (float): Volatilité annuelle du sous-jacent.
    N (int): Nombre de pas dans l'arbre binomial.
    exercise_type (str): 'EU' pour Européenne, 'US' pour Américaine.
    discrete_dividends (list): Liste de tuples (montant_dividende, temps_dividende).
                               Ex: [(D1, T_div1), (D2, T_div2)]. Temps en années.

    Retourne:
    float: Le prix de l'option.
    """
    if discrete_dividends is None:
        discrete_dividends = []

    # Sort the dividends by time, crucial for correct handling
    discrete_dividends.sort(key=lambda x: x[1])

    dt = T / N  # Durée de chaque pas
    u = np.exp(sigma * np.sqrt(dt))  # Facteur de hausse
    d = 1 / u  # Facteur de baisse
    p = (np.exp(r * dt) - d) / (u - d)  # Probabilité neutre au risque
    q = 1 - p  # Probabilité de baisse

    # --- Étape 1 : Initialisation de l'arbre des prix du sous-jacent ---
    S_tree = np.zeros((N + 1, N + 1))
    option_values = np.zeros((N + 1, N + 1))

    # Calcul du prix spot "ajusté" pour les dividendes
    # Pour les options américaines, il est plus simple de projeter S_0 pour chaque dividende
    # et ajuster localement les valeurs de l'arbre.
    # Pour le modèle Binomial CRR, S_0 est le prix du sous-jacent à l'instant t=0.
    # La gestion des dividendes se fera en ajustant les prix aux noeuds de l'arbre
    # juste après le détachement de chaque dividende.

    # Initialisation du noeud de départ
    S_tree[0, 0] = S

    # Points dans le temps où les dividendes sont payés (en termes de 'pas' de l'arbre)
    # Convertir les temps de dividende en indices de pas pour l'arbre
    dividend_steps = []
    for div_amount, div_time in discrete_dividends:
        # Trouver le pas immédiatement après ou au moment du dividende
        step_idx = int(np.floor(div_time / dt))
        if step_idx >= N:  # Dividende est à l'échéance ou après
            continue
        dividend_steps.append((step_idx, div_amount))

    # Construire l'arbre des prix
    for i in range(1, N + 1):  # Pour chaque pas de temps
        # Vérifier si un dividende est payé à ce pas 'i' (ou juste avant)
        current_dividend_amount = 0.0
        # Check if there's a dividend ex-date at or just before this step
        # Assuming dividend is paid at the end of the step it falls into,
        # or at step_idx if div_time / dt is exactly step_idx.
        # For simplicity, we assume dividend is paid *after* the stock moves for this step.
        # So stock price *before* dividend is S_tree[i-1,j]*u or S_tree[i-1,j]*d.
        # Then, if a dividend is paid, the stock price drops by D.
        # The stock price at time (i*dt) is S_i.
        
        # We need to consider dividends that occur between (i-1)*dt and i*dt.
        # It's generally handled by looking at dividend times.
        # The dividend happens *after* the up/down move, at the ex-dividend date.
        # So we adjust the stock price *at the dividend step*.

        # Store stock prices *before* dividend at this step
        S_tree_before_dividend_at_step_i = np.zeros(i + 1)

        for j in range(i + 1):  # Pour chaque nœud à ce pas
            if j == 0:
                # Premier noeud à ce pas (tout en bas)
                S_tree_before_dividend_at_step_i[j] = S_tree[i - 1, 0] * d
            else:
                # Autres noeuds
                S_tree_before_dividend_at_step_i[j] = S_tree[i - 1, j - 1] * u

        # Apply dividend adjustments if any dividend ex-date falls within this step interval (or at this step's end)
        for div_step_idx, div_amount in dividend_steps:
            if div_step_idx == i - 1: # The dividend payment falls into the current interval (i-1)*dt to i*dt
                # This means at the *beginning* of step i, the stock drops by div_amount
                # Or, more precisely, after the calculation of S_tree[i,j] based on S_tree[i-1,j-1]*u or S_tree[i-1,0]*d
                # the price then drops by div_amount *at this node*.
                # This is a common simplification for discrete dividends in binomial trees.
                
                # So we apply the dividend adjustment *after* the multiplicative step,
                # effectively reducing the stock price for the nodes *at* step `i`
                # (which represents the stock price at time `i*dt` AFTER dividend payment).
                for j in range(i + 1):
                    S_tree[i, j] = max(0, S_tree_before_dividend_at_step_i[j] - div_amount)
                break # Assuming one dividend per step for simplicity or sort and apply all for this step
        else: # No dividend at this exact step, just apply normal pricing
             for j in range(i + 1):
                S_tree[i, j] = S_tree_before_dividend_at_step_i[j]

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

            # Décision d'exercice (pour options Américaines)
            if exercise_type == "US":
                option_values[i, j] = max(continuation_value, intrinsic_value)
            else:  # Pour les options Européennes, pas d'exercice anticipé
                option_values[i, j] = continuation_value

    return option_values[0, 0]