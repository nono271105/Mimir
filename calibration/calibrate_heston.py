# mimir/calibration/calibrate_heston.py
import yfinance as yf
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize
import numpy as np

# IMPORT MIS À JOUR pour utiliser votre market_data_loader.py
from data.market_data_loader import (
    get_current_stock_price,
    get_option_chain,
    get_risk_free_rate,
)
from calibration.objective_function import heston_objective_function


def run_heston_calibration(
    ticker_symbol: str,
    expiration_date_str: str,
    initial_params: tuple,
    bounds: tuple,
):
    print(f"\n--- Démarrage de la calibration Heston pour {ticker_symbol} ---")

    # Récupération des données de marché
    spot_price = get_current_stock_price(ticker_symbol)
    if spot_price is None:
        print(f"Erreur: Impossible de récupérer le prix spot pour {ticker_symbol}.")
        return {"status": "failed", "message": "Failed to fetch spot price."}
    print(f"Prix Spot actuel ({ticker_symbol}): {spot_price}")

    # Utilisation de get_option_chain et combinaison des résultats
    calls_df, puts_df = get_option_chain(ticker_symbol, expiration_date_str)

    if not calls_df.empty:
        calls_df["optionType"] = "call"
    if not puts_df.empty:
        puts_df["optionType"] = "put"

    market_options_df = pd.concat([calls_df, puts_df], ignore_index=True)

    # AJOUT ESSENTIEL : Ajouter la colonne 'expiration' au DataFrame
    # Cette colonne est nécessaire pour objective_function.py
    market_options_df['expiration'] = expiration_date_str


    if market_options_df.empty:
        print(
            f"Erreur: Aucune donnée d'option valide trouvée pour {expiration_date_str}."
        )
        return {"status": "failed", "message": "No valid options data."}

    # Filtrer les options avec des prix bid/ask valides
    market_options_df = market_options_df.dropna(subset=["bid", "ask"])
    market_options_df = market_options_df[
        (market_options_df["bid"] > 0) & (market_options_df["ask"] > 0)
    ]
    if market_options_df.empty:
        print(f"Aucune option avec des prix bid/ask valides trouvée.")
        return {"status": "failed", "message": "No valid bid/ask prices."}

    print(
        f"Nombre d'options de marché valides pour la calibration: {len(market_options_df)}"
    )

    # Récupération du taux sans risque
    risk_free_rate = get_risk_free_rate() # get_risk_free_rate retourne déjà en décimal
    print(f"Taux sans risque (via ^TNX) récupéré: {risk_free_rate * 100:.2f}%")
    print(f"Taux sans risque utilisé: {risk_free_rate * 100:.2f}%")

    current_date = datetime.now()
    print(f"Date de calibration: {current_date.strftime('%Y-%m-%d')}")

    print("\nDémarrage de l'optimisation...")

    result = minimize(
        lambda params: heston_objective_function(
            params, market_options_df, spot_price, risk_free_rate, current_date
        ),
        initial_params,
        method="L-BFGS-B",
        bounds=bounds,
        options={"disp": True, "maxiter": 10000},  # Affiche le processus, max 10000 itérations
    )

    # Afficher les résultats
    calibrated_params = result.x
    min_error = result.fun

    print("\n--- Résultats de la Calibration ---")
    if result.success:
        print("Calibration réussie!")
    else:
        print(f"Calibration terminée, mais la convergence n'est pas garantie. Message: {result.message}")

    print("Paramètres Heston calibrés (V0, kappa, theta, xi, rho):")
    print(f"  V0: {calibrated_params[0]:.6f}")
    print(f"  kappa: {calibrated_params[1]:.6f}")
    print(f"  theta: {calibrated_params[2]:.6f}")
    print(f"  xi: {calibrated_params[3]:.6f}")
    print(f"  rho: {calibrated_params[4]:.6f}")
    print(f"Erreur minimale (Somme des erreurs quadratiques): {min_error:.6f}")

    return {
        "status": "success" if result.success else "failed",
        "calibrated_params": calibrated_params.tolist(),
        "min_error": min_error,
        "optimizer_message": result.message,
    }





if __name__ == "__main__":
    ticker = "AAPL"
    # Utiliser une date d'expiration future proche pour avoir des options liquides
    expiration = "2025-07-18" # J'ai mis une date un peu plus lointaine pour plus de chance d'avoir des options
                              # Vous pouvez ajuster cette date si besoin.

    # Paramètres initiaux pour Heston (V0, kappa, theta, xi, rho)
    initial_params_heston = (0.05, 1.8, 0.045, 0.6, -0.5)

    # Bornes pour les paramètres Heston
    bounds_heston = (
        (1e-5, 1.0),   # V0 (variance initiale)
        (1e-5, 10.0),    # kappa (vitesse de retour à la moyenne)
        (1e-5, 1.0),   # theta (variance à long terme)
        (1e-5, 5.0),    # xi (volatilité de la volatilité)
        (-0.99, 0.99),  # rho (corrélation)
    )

    calibration_results = run_heston_calibration(
        ticker_symbol=ticker,
        expiration_date_str=expiration,
        initial_params=initial_params_heston,
        bounds=bounds_heston,
    )
    print(calibration_results)