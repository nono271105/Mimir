# main.py
from src.models.bsm_model import black_scholes_greeks
from src.models.binomial_model import binomial_option_pricing
from src.ui.cli_interface import get_user_inputs_common # Renommé
from src.ui.display_results import display_bsm_results, plot_payoff

def main():
    """
    Fonction principale de l'application Mimir.
    Orchestre la demande des inputs, le calcul d'options et l'affichage des résultats.
    """
    print("Bienvenue dans Mimir : Le Calculateur d'Options")

    # 1. Demander à l'utilisateur le type d'option (Européenne/Américaine)
    option_exercise_type = ""
    while option_exercise_type not in ['EU', 'US']:
        option_exercise_type = input("Quel type d'option souhaitez-vous calculer ? (EU pour Européenne, US pour Américaine) : ").upper()
        if option_exercise_type not in ['EU', 'US']:
            print("Choix invalide. Veuillez entrer 'EU' ou 'US'.")

    # 2. Demander les paramètres communs de l'option
    params = get_user_inputs_common() # Appel à la fonction renommée
    
    # Extraire les paramètres du dictionnaire
    option_type = params['option_type'] # C ou P
    S = params['S']
    K = params['K']
    T = params['T']
    r = params['r']
    sigma = params['sigma']

    # Initialisation des paramètres de dividendes
    dividend_yield = 0.0      # Pour BSM
    dividend_amount = 0.0     # Pour Binomial (discret)
    dividend_time = -1.0      # Pour Binomial (discret)

    option_price = 0.0 # Initialisation du prix de l'option
    
    if option_exercise_type == 'EU':
        print("\n--- Modèle utilisé : Black-Scholes-Merton (BSM) ---")
        # Demander le rendement des dividendes CONTINUS SPÉCIFIQUEMENT pour BSM
        while True:
            try:
                dividend_yield = float(input("Entrez le rendement annuel des dividendes (q, ex: 0.01 pour 1%, 0 si pas de dividendes) : "))
                if dividend_yield < 0:
                    print("Le rendement des dividendes ne peut pas être négatif.")
                else:
                    break
            except ValueError:
                print("Entrée invalide. Veuillez entrer un nombre.")

        # Calcul avec Black-Scholes, en passant le rendement des dividendes
        option_price, d1_val, d2_val, N_d1_val, N_d2_val, \
        delta_val, gamma_val, vega_val, theta_val, rho_val = \
            black_scholes_greeks(option_type, S, K, T, r, sigma, dividend_yield)
        
        # Afficher les résultats textuels spécifiques à BSM (avec Grecs)
        display_bsm_results(option_type, option_price, d1_val, d2_val, N_d1_val, N_d2_val,
                            delta_val, gamma_val, vega_val, theta_val, rho_val)

    elif option_exercise_type == 'US':
        print("\n--- Modèle utilisé : Binomial (pour options Américaines) ---")
        
        # Demander le nombre de pas pour le modèle binomial
        N_steps = 0
        while N_steps <= 0:
            try:
                N_steps = int(input("Entrez le nombre de pas pour le modèle binomial (N) : "))
                if N_steps <= 0:
                    print("Le nombre de pas doit être un entier positif.")
                else:
                    break
            except ValueError:
                print("Entrée invalide. Veuillez entrer un nombre entier.")
        
        # --- Demander des informations sur les dividendes DISCRETS SPÉCIFIQUEMENT pour le modèle Binomial ---
        while True:
            has_dividends = input("Y a-t-il un dividende discret à prendre en compte pour l'option Américaine ? (oui/non) : ").lower()
            if has_dividends == 'oui':
                while True:
                    try:
                        dividend_amount = float(input("Entrez le montant du dividende discret ($) : "))
                        if dividend_amount < 0:
                            print("Le montant du dividende ne peut pas être négatif.")
                        else:
                            break
                    except ValueError:
                        print("Entrée invalide. Veuillez entrer un nombre.")
                
                while True:
                    try:
                        # Le temps du dividende doit être avant l'échéance mais après l'instant t=0
                        dividend_time = float(input(f"Entrez le temps jusqu'au dividende (T_div, doit être > 0 et < {T}) : "))
                        if not (0 < dividend_time < T):
                            print(f"Le temps jusqu'au dividende doit être strictement entre 0 et le temps à l'échéance ({T}).")
                        else:
                            break
                    except ValueError:
                        print("Entrée invalide. Veuillez entrer un nombre.")
                break # Sortir de la boucle has_dividends
            elif has_dividends == 'non':
                break # Sortir de la boucle has_dividends
            else:
                print("Réponse invalide. Veuillez répondre 'oui' ou 'non'.")

        # Calcul avec le modèle Binomial, en passant 'US' et les paramètres de dividende discret
        try:
            option_price = binomial_option_pricing(option_type, S, K, T, r, sigma, N_steps, 
                                                   exercise_type='US', 
                                                   dividend_yield=0.0, # Rendement continu non pertinent pour dividendes discrets, mais gardé pour la signature
                                                   dividend_amount=dividend_amount, 
                                                   dividend_time=dividend_time)
            print(f"\n--- Résultats du Modèle Binomial Américain (N={N_steps}) ---")
            print(f"Le prix de l'option {option_type} est : {option_price:.2f} $")
        except ValueError as e:
            print(f"Erreur lors du calcul binomial: {e}. Veuillez vérifier vos paramètres.")
            option_price = 0.0
        
    if option_price > 0.0:
        plot_payoff(option_type, S, K, option_price)

    print("\nCalcul terminé. Au revoir de Mimir !")

if __name__ == "__main__":
    main()