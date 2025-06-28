# mimir/calibration/objective_function.py
import numpy as np
import pandas as pd
from src.models.european.pricing import price_heston_european_option
from datetime import datetime


# Helper to calculate time to maturity in years from expiration date string
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
                        Note: V0 sera ici la variance initiale calibrée.
        market_options_df (pd.DataFrame): DataFrame contenant les options de marché
                                           avec 'strike', 'optionType', 'expiration', 'bid', 'ask'.
        S0 (float): Prix spot actuel du sous-jacent.
        r (float): Taux sans risque.
        current_date (datetime): Date actuelle pour le calcul du TTM.
        # Les paramètres N_steps_mc et N_simulations_mc sont supprimés car Monte Carlo n'est plus utilisé.

    Returns:
        float: La somme des erreurs quadratiques (Mean Squared Error - MSE).
    """
    V0, kappa, theta, xi, rho = params

    if not (V0 >= 0 and kappa >= 0 and theta >= 0 and xi >= 0 and -1 <= rho <= 1):
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

        if pd.isna(market_bid) or pd.isna(market_ask) or market_ask == 0:
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
        except Exception as e:
            print(f"Erreur de pricing pour K={K}, T={T}, type={option_type}: {e}")
            return np.inf

    if num_options == 0:
        return np.inf

    return total_squared_error