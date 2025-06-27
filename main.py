# main.py
from src.models.bsm_model import black_scholes_greeks
from src.ui.cli_interface import get_user_inputs
from src.ui.display_results import display_bsm_results, plot_payoff

def main():
    """
    Fonction principale de l'application Mimir.
    Orchestre la demande des inputs, le calcul Black-Scholes et l'affichage des résultats.
    """
    print("Bienvenue dans Mimir : Le Calculateur d'Options Black-Scholes")

    # 1. Demander les paramètres à l'utilisateur
    params = get_user_inputs()

    # Extraire les paramètres du dictionnaire pour le passage à la fonction de calcul
    option_type = params['option_type']
    S = params['S']
    K = params['K']
    T = params['T']
    r = params['r']
    sigma = params['sigma']

    # 2. Calculer le prix de l'option et les Grecs en utilisant le modèle BSM
    option_price, d1_val, d2_val, N_d1_val, N_d2_val, \
    delta_val, gamma_val, vega_val, theta_val, rho_val = \
        black_scholes_greeks(option_type, S, K, T, r, sigma)

    # 3. Afficher les résultats textuels
    display_bsm_results(option_type, option_price, d1_val, d2_val, N_d1_val, N_d2_val,
                        delta_val, gamma_val, vega_val, theta_val, rho_val)

    # 4. Afficher le graphique du profit/perte net
    plot_payoff(option_type, S, K, option_price)

    print("\nCalcul terminé. Au revoir de Mimir !")

if __name__ == "__main__":
    main()