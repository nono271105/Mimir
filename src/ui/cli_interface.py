# src/ui/cli_interface.py
def get_user_inputs_common(): # Renommée pour indiquer "inputs communs"
    """
    Demande à l'utilisateur les paramètres communs nécessaires pour le calcul de l'option.
    Effectue une validation de base des entrées.

    Returns:
        dict: Un dictionnaire contenant les paramètres valides communs.
    """
    params = {}

    while True:
        option_type_input = input("Voulez-vous calculer le prix d'une option Call (C) ou Put (P) ? ").upper()
        if option_type_input in ['C', 'P']:
            params['option_type'] = option_type_input
            break
        else:
            print("Entrée invalide. Veuillez entrer 'C' pour Call ou 'P' pour Put.")

    while True:
        try:
            S = float(input("Entrez le prix spot actuel (S) : "))
            if S <= 0:
                raise ValueError("Le prix spot doit être positif.")
            params['S'] = S
            break
        except ValueError:
            print("Entrée invalide. Veuillez entrer un nombre positif pour le prix spot.")

    while True:
        try:
            K = float(input("Entrez le prix d'exercice (K) : "))
            if K <= 0:
                raise ValueError("Le prix d'exercice doit être positif.")
            params['K'] = K
            break
        except ValueError:
            print("Entrée invalide. Veuillez entrer un nombre positif pour le prix d'exercice.")

    while True:
        try:
            T = float(input("Entrez le temps jusqu'à l'échéance en années (T) : "))
            if T < 0: # Peut être 0 pour l'échéance
                raise ValueError("Le temps jusqu'à l'échéance ne peut pas être négatif.")
            params['T'] = T
            break
        except ValueError:
            print("Entrée invalide. Veuillez entrer un nombre positif ou nul pour le temps.")

    while True:
        try:
            r = float(input("Entrez le taux d'intérêt sans risque (r, ex: 0.045 pour 4.5%) : "))
            params['r'] = r
            break
        except ValueError:
            print("Entrée invalide. Veuillez entrer un nombre pour le taux d'intérêt.")

    while True:
        try:
            sigma = float(input("Entrez la volatilité (sigma, ex: 0.2 pour 20%) : "))
            if sigma < 0: # sigma = 0 est géré par le modèle, mais pas négatif
                raise ValueError("La volatilité doit être positive ou nulle.")
            params['sigma'] = sigma
            break
        except ValueError:
            print("Entrée invalide. Veuillez entrer un nombre positif ou nul pour la volatilité.")
            
            
    return params