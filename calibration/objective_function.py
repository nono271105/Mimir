# mimir/calibration/objective_function.py
import numpy as np
import pandas as pd
from src.models.european.pricing import price_heston_european_option
from datetime import datetime


# Helper pour calculer le temps à maturité en années
def calculate_T_from_expiration(
    expiration_date_str: str, current_date: datetime
) -> float:
    """
    Calcule le temps jusqu'à l'échéance en années à partir d'une date d'expiration.
    """
    expiration_date = datetime.strptime(expiration_date_str, "%Y-%m-%d")
    diff_days = (expiration_date - current_date).days
    return diff_days / 365.0


def heston_objective_function(
    params: tuple,
    market_options_df: pd.DataFrame,
    S0: float,
    r: float,
    current_date: datetime,
) -> float:
    """
    Fonction objectif pour la calibration du modèle de Heston.
    Elle calcule la somme des erreurs quadratiques entre les prix du modèle
    et les prix de marché des options.

    Paramètres:
        params (tuple): Tuple des paramètres de Heston (V0, kappa, theta, xi, rho).
        market_options_df (pd.DataFrame): DataFrame contenant les options de marché
                                           avec 'strike', 'optionType', 'expiration', 'bid', 'ask'.
        S0 (float): Prix spot actuel du sous-jacent.
        r (float): Taux sans risque.
        current_date (datetime): Date actuelle pour le calcul du TTM.

    Returns:
        float: La somme des erreurs quadratiques.
    """
    V0, kappa, theta, xi, rho = params

    # --- Débogage : Afficher les paramètres testés par l'optimiseur ---
    #print(f"\nDEBUG: Testing params: V0={V0:.6f}, kappa={kappa:.6f}, theta={theta:.6f}, xi={xi:.6f}, rho={rho:.6f}")
    # --- Fin Débogage ---

    # Vérification des contraintes "douces" sur les paramètres (en plus des bornes de minimize)
    # Retourne une erreur infinie si les paramètres sont hors de plages physiquement acceptables
    if not (V0 > 0 and kappa > 0 and theta > 0 and xi > 0 and -1 <= rho <= 1):
        return np.inf 

    total_squared_error = 0.0
    num_options = 0

    for index, row in market_options_df.iterrows():
        K = row["strike"]
        option_type_str = row["optionType"]
        option_type = "C" if option_type_str.lower() == "call" else "P"

        expiration_date_str = row["expiration"]
        T = calculate_T_from_expiration(expiration_date_str, current_date)

        market_bid = row["bid"]
        market_ask = row["ask"]
        
        # Filtre additionnel au cas où bid/ask seraient invalides ici
        if pd.isna(market_bid) or pd.isna(market_ask) or market_ask == 0 or market_bid == 0:
            continue

        market_price = (market_bid + market_ask) / 2.0

        try:
            model_price = price_heston_european_option(
                S0=S0,
                V0=V0,
                kappa=kappa,
                theta=theta,
                xi=xi,
                rho=rho,
                T=T,
                r=r,
                K=K,
                option_type=option_type,
            )
            squared_error = (model_price - market_price) ** 2
            total_squared_error += squared_error
            num_options += 1

            # --- Débogage : Afficher les prix individuels et l'erreur ---
            #print(f"  Option K={K}, T={T:.2f}, Type={option_type}: Market={market_price:.4f}, Model={model_price:.4f}, Error^2={squared_error:.4f}")
            # --- Fin Débogage ---

        except Exception as e:
            # Si une erreur de pricing survient pour une option, on peut retourner infini
            # ou juste ignorer cette option et continuer. Pour la calibration, retourner infini
            # est souvent préféré car cela pénalise les paramètres qui causent l'erreur.
            # print(f"DEBUG: Erreur de pricing pour K={K}, T={T}, type={option_type}: {e}")
            return np.inf # Indique à l'optimiseur que ces paramètres sont "mauvais"

    if num_options == 0:
        # print("DEBUG: Aucune option traitée, retour de np.inf")
        return np.inf

    # --- Débogage : Afficher l'erreur totale pour ce set de paramètres ---
    #print(f"DEBUG: Total Squared Error for this set of params: {total_squared_error:.4f}")
    # --- Fin Débogage ---

    return total_squared_error