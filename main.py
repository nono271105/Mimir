# main.py
from src.models.bsm_model import black_scholes_greeks
from src.models.binomial_model import binomial_option_pricing
from src.ui.cli_interface import get_user_inputs
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

    # 2. Demander les paramètres de l'option
    params = get_user_inputs()

    # Extraire les paramètres du dictionnaire
    option_type = params['option_type'] # C ou P
    S = params['S']
    K = params['K']
    T = params['T']
    r = params['r']
    sigma = params['sigma']

    option_price = 0.0 # Initialisation du prix de l'option

    if option_exercise_type == 'EU':
        print("\n--- Modèle utilisé : Black-Scholes-Merton (BSM) ---")
        # Calcul avec Black-Scholes
        option_price, d1_val, d2_val, N_d1_val, N_d2_val, \
        delta_val, gamma_val, vega_val, theta_val, rho_val = \
            black_scholes_greeks(option_type, S, K, T, r, sigma)

        # Afficher les résultats textuels spécifiques à BSM (avec Grecs)
        display_bsm_results(option_type, option_price, d1_val, d2_val, N_d1_val, N_d2_val,
                            delta_val, gamma_val, vega_val, theta_val, rho_val)

    elif option_exercise_type == 'US':
        print("\n--- Modèle utilisé : Binomial (pour options Américaines - implémentation en cours) ---")
        # Demander le nombre de pas pour le modèle binomial
        N_steps = 0
        while N_steps <= 0:
            try:
                N_steps = int(input("Entrez le nombre de pas pour le modèle binomial (N) : "))
                if N_steps <= 0:
                    print("Le nombre de pas doit être un entier positif.")
            except ValueError:
                print("Entrée invalide. Veuillez entrer un nombre entier.")

        # Calcul avec le modèle Binomial
        try:
            # IMPORTANT: Notre modèle binomial actuel (binomial_option_pricing) calcule des options européennes. 
            # Pour les options américaines, il faudra ajouter la logique d'exercice anticipé dans ce modèle dans une phase future.
            # Pour l'instant, il donnera le prix d'une EU.
            option_price = binomial_option_pricing(option_type, S, K, T, r, sigma, N_steps)
            print(f"\n--- Résultats du Modèle Binomial (N={N_steps}) ---")
            print(f"Le prix de l'option {option_type} est : {option_price:.2f} $")
            
            # Note: Les Grecs ne sont pas calculés directement par cette implémentation binomiale
        except ValueError as e:
            print(f"Erreur lors du calcul binomial: {e}. Veuillez vérifier vos paramètres.")
            option_price = 0.0 # Réinitialiser le prix en cas d'erreur

    if option_price > 0.0: # Afficher le graphique seulement si le calcul a réussi et le prix est valide
        # Afficher le graphique du profit/perte net
        # Le graphique de payoff est toujours le même quelle que soit la méthode de calcul
        plot_payoff(option_type, S, K, option_price)

    print("\nCalcul terminé. Au revoir de Mimir !")

if __name__ == "__main__":
    main()