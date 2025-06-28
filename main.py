# main.py
from src.models.bsm_model import black_scholes_greeks
from src.models.binomial_model import binomial_option_pricing
from src.models.bjerksund_stensland_model import (
    bjerksund_stensland_2002,
)  # Nouvelle importation
from src.ui.cli_interface import get_user_inputs_common
from src.ui.display_results import display_bsm_results, plot_payoff


def main():
    """
    Fonction principale de l'application Mimir.
    Orchestre la demande des inputs, le calcul d'options et l'affichage des résultats.
    """
    print("Bienvenue dans Mimir : Le Calculateur d'Options")

    # 1. Demander à l'utilisateur le type d'option (Européenne/Américaine)
    option_exercise_type = ""
    while option_exercise_type not in ["EU", "US"]:
        option_exercise_type = input(
            "Quel type d'option souhaitez-vous calculer ? (EU pour Européenne, US pour Américaine) : "
        ).upper()
        if option_exercise_type not in ["EU", "US"]:
            print("Choix invalide. Veuillez entrer 'EU' ou 'US'.")

    # 2. Demander les paramètres communs de l'option
    params = get_user_inputs_common()

    # Extraire les paramètres du dictionnaire
    option_type = params["option_type"]  # C ou P
    S = params["S"]
    K = params["K"]
    T = params["T"]
    r = params["r"]
    sigma = params["sigma"]

    # Initialisation des paramètres de dividendes
    dividend_yield = 0.0  # Pour BSM et Bjerksund-Stensland (rendement continu)
    discrete_dividends = []  # Pour Binomial (discrets)

    option_price = 0.0  # Initialisation du prix de l'option

    if option_exercise_type == "EU":
        print("\n--- Modèle utilisé : Black-Scholes-Merton (BSM) ---")
        # Demander le rendement des dividendes CONTINUS SPÉCIFIQUEMENT pour BSM
        while True:
            try:
                dividend_yield = float(
                    input(
                        "Entrez le rendement annuel des dividendes (q, ex: 0.01 pour 1%, 0 si pas de dividendes) : "
                    )
                )
                if dividend_yield < 0:
                    print("Le rendement des dividendes ne peut pas être négatif.")
                else:
                    break
            except ValueError:
                print("Erreur: Entrée invalide. Veuillez entrer un nombre.")

        # Calcul avec Black-Scholes, en passant le rendement des dividendes
        (
            option_price,
            d1_val,
            d2_val,
            N_d1_val,
            N_d2_val,
            delta_val,
            gamma_val,
            vega_val,
            theta_val,
            rho_val,
        ) = black_scholes_greeks(option_type, S, K, T, r, sigma, dividend_yield)

        # Afficher les résultats textuels spécifiques à BSM (avec Grecs)
        display_bsm_results(
            option_type,
            option_price,
            d1_val,
            d2_val,
            N_d1_val,
            N_d2_val,
            delta_val,
            gamma_val,
            vega_val,
            theta_val,
            rho_val,
        )

    elif option_exercise_type == "US":
        # Demander à l'utilisateur quel modèle utiliser pour les options américaines
        american_model_choice = ""
        while american_model_choice not in ["BINOMIAL", "BS"]:
            american_model_choice = input(
                "Quel modèle souhaitez-vous utiliser pour les options Américaines ? (BINOMIAL ou BS pour Bjerksund-Stensland) : "
            ).upper()
            if american_model_choice not in ["BINOMIAL", "BS"]:
                print("Choix invalide. Veuillez entrer 'BINOMIAL' ou 'BS'.")

        if american_model_choice == "BINOMIAL":
            print("\n--- Modèle utilisé : Binomial (pour options Américaines) ---")

            # Demander le nombre de pas pour le modèle binomial
            N_steps = 0
            while N_steps <= 0:
                try:
                    N_steps = int(
                        input(
                            "Entrez le nombre de pas pour le modèle binomial (N > 0) : "
                        )
                    )
                    if N_steps <= 0:
                        print("Erreur: Le nombre de pas doit être un entier positif.")
                    else:
                        break
                except ValueError:
                    print("Erreur: Entrée invalide. Veuillez entrer un nombre entier.")

            # --- Demander des informations sur les dividendes discrets pour le modèle Binomial ---
            while True:
                has_dividends = input(
                    "Y a-t-il des dividendes discrets à prendre en compte pour l'option Américaine ? (oui/non) : "
                ).lower()
                if has_dividends == "oui":
                    num_dividends = 0
                    while True:
                        try:
                            num_dividends = int(
                                input(
                                    "Combien de dividendes discrets (max 4, 0 pour annuler) ? "
                                )
                            )
                            if 0 <= num_dividends <= 4:
                                break
                            else:
                                print(
                                    "Erreur: Le nombre de dividendes doit être entre 0 et 4."
                                )
                        except ValueError:
                            print(
                                "Erreur: Entrée invalide. Veuillez entrer un nombre entier."
                            )

                    for i in range(num_dividends):
                        while True:
                            try:
                                dividend_amount = float(
                                    input(
                                        f"Entrez le montant du dividende #{i+1} (D, en $) : "
                                    )
                                )
                                if dividend_amount < 0:
                                    print(
                                        "Erreur: Le montant du dividende ne peut pas être négatif."
                                    )
                                    continue
                                break
                            except ValueError:
                                print(
                                    "Erreur: Entrée invalide. Veuillez entrer un nombre."
                                )

                        while True:
                            try:
                                dividend_days = int(
                                    input(
                                        f"Dans combien de jours aura lieu le dividende #{i+1} ? "
                                    )
                                )
                                if dividend_days <= 0:
                                    print(
                                        "Erreur: Le nombre de jours doit être positif."
                                    )
                                    continue

                                dividend_time_in_years = dividend_days / 365.0

                                if not (0 < dividend_time_in_years < T):
                                    print(
                                        f"Erreur: Le dividende doit avoir lieu STRICTEMENT entre la date d'aujourd'hui et la date d'échéance ({T*365:.0f} jours)."
                                    )
                                    continue
                                break
                            except ValueError:
                                print(
                                    "Erreur: Entrée invalide. Veuillez entrer un nombre entier de jours."
                                )

                        discrete_dividends.append(
                            (dividend_amount, dividend_time_in_years)
                        )
                    break
                elif has_dividends == "non":
                    break
                else:
                    print("Réponse invalide. Veuillez répondre 'oui' ou 'non'.")

            # Calcul avec le modèle Binomial, en passant 'US' et les paramètres de dividendes discrets
            try:
                option_price = binomial_option_pricing(
                    option_type,
                    S,
                    K,
                    T,
                    r,
                    sigma,
                    N_steps,
                    exercise_type="US",
                    discrete_dividends=discrete_dividends,  # Passer la liste des dividendes
                )
                print(f"\n--- Résultats du Modèle Binomial Américain (N={N_steps}) ---")
                print(f"Le prix de l'option {option_type} est : {option_price:.2f} $")
            except ValueError as e:
                print(
                    f"Erreur lors du calcul binomial: {e}. Veuillez vérifier vos paramètres."
                )
                option_price = 0.0

        elif american_model_choice == "BS":
            print("\n--- Modèle utilisé : Bjerksund-Stensland (BS) ---")
            # Demander le rendement des dividendes CONTINUS pour Bjerksund-Stensland
            while True:
                try:
                    dividend_yield = float(
                        input(
                            "Entrez le rendement annuel des dividendes (q, ex: 0.01 pour 1%, 0 si pas de dividendes) : "
                        )
                    )
                    if dividend_yield < 0:
                        print("Le rendement des dividendes ne peut pas être négatif.")
                    else:
                        break
                except ValueError:
                    print("Erreur: Entrée invalide. Veuillez entrer un nombre.")

            # Calcul avec Bjerksund-Stensland
            try:
                option_price = bjerksund_stensland_2002(
                    option_type, S, K, T, r, sigma, q=dividend_yield
                )
                print(f"\n--- Résultats du Modèle Bjerksund-Stensland Américain ---")
                print(f"Le prix de l'option {option_type} est : {option_price:.2f} $")
            except ValueError as e:
                print(
                    f"Erreur lors du calcul Bjerksund-Stensland: {e}. Veuillez vérifier vos paramètres."
                )
                option_price = 0.0

    if option_price > 0.0:
        plot_payoff(option_type, S, K, option_price)

    print("\nCalcul terminé. Au revoir de Mimir !")


if __name__ == "__main__":
    main()
