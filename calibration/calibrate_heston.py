# mimir/calibration/calibrate_heston.py

import yfinance as yf
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize
import numpy as np

# IMPORT MIS À JOUR
from data.market_data_loader import (
    get_current_stock_price,
    get_option_chain,
    get_risk_free_rate,
    get_option_expirations # Ajout de cette fonction pour récupérer les expirations
)
from calibration.objective_function import heston_objective_function


# --- Variables globales pour la progression de l'optimisation ---
# Ceci permettra de suivre l'état de la progression
current_iteration = 0
total_iterations_expected = 2000 # Doit correspondre à 'maxiter' dans minimize options

def optimization_callback(xk):
    """
    Fonction de rappel (callback) appelée après chaque itération de l'optimiseur.
    xk est le vecteur des paramètres à l'itération courante.
    """
    global current_iteration
    current_iteration += 1
    
    # Calcul du pourcentage de progression
    percentage = (current_iteration / total_iterations_expected) * 100
    
    # Affichage de la barre de progression (similaire à pip)
    # '\r' ramène le curseur au début de la ligne
    # 'end=''' empêche un saut de ligne
    # 'flush=True' assure que la sortie est affichée immédiatement
    print(f'\rOptimisation en cours... {current_iteration}/{total_iterations_expected} itérations ({percentage:.2f}%)', end='', flush=True)

# --- Fin des variables et fonction de progression ---


def run_heston_calibration(
    ticker_symbol: str,
    expiration_date_str: str,
    initial_params: tuple,
    bounds: tuple,
):
    global current_iteration # Assurez-vous que la fonction peut réinitialiser le compteur
    current_iteration = 0 # Réinitialise le compteur à chaque nouvelle calibration

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

    # AJOUT ESSENTIEL : Ajouter la colonne 'expiration' au DataFrame (correction du KeyError)
    market_options_df['expiration'] = expiration_date_str

    if market_options_df.empty:
        print(
            f"Erreur: Aucune donnée d'option valide trouvée pour {expiration_date_str}."
        )
        return {"status": "failed", "message": "No valid options data."}

    # FILTRAGE AVANCÉ DES DONNÉES DE MARCHÉ (très important pour la calibration)
    print(f"Nombre d'options brutes: {len(market_options_df)}")

    # Assurez-vous que les colonnes 'bid' et 'ask' sont numériques et non nulles avant le filtrage
    market_options_df['bid'] = pd.to_numeric(market_options_df['bid'], errors='coerce')
    market_options_df['ask'] = pd.to_numeric(market_options_df['ask'], errors='coerce')
    market_options_df = market_options_df.dropna(subset=['bid', 'ask'])
    market_options_df = market_options_df[(market_options_df['bid'] > 0) & (market_options_df['ask'] > 0)].copy()


    # 1. Filtrer les options par leur "Moneyness" (proximité du strike par rapport au spot)
    percentage_threshold = 0.35 
    min_strike = spot_price * (1 - percentage_threshold)
    max_strike = spot_price * (1 + percentage_threshold)
    market_options_df = market_options_df[
        (market_options_df['strike'] >= min_strike) &
        (market_options_df['strike'] <= max_strike)
    ].copy() 

    # 2. Filtrer par Spread Bid-Ask (indicateur de liquidité)
    max_spread_percentage = 0.20 
    market_options_df['spread_pct'] = (market_options_df['ask'] - market_options_df['bid']) / market_options_df['bid']
    market_options_df = market_options_df[market_options_df['spread_pct'] <= max_spread_percentage].copy()
    market_options_df = market_options_df.drop(columns=['spread_pct'])

    # Filter by Open Interest or Volume for better data quality
    market_options_df['volume'] = pd.to_numeric(market_options_df['volume'], errors='coerce').fillna(0)
    market_options_df['openInterest'] = pd.to_numeric(market_options_df['openInterest'], errors='coerce').fillna(0)

    market_options_df = market_options_df[
        (market_options_df['volume'] > 0) | (market_options_df['openInterest'] > 0)
    ].copy()


    if market_options_df.empty:
        print(f"Erreur: Aucune option de marché valide après filtrage strict. Adoucissez les filtres.")
        return {"status": "failed", "message": "No valid options after filtering."}


    print(
        f"Nombre d'options de marché valides après filtrage: {len(market_options_df)}"
    )

    # Récupération du taux sans risque
    risk_free_rate = get_risk_free_rate() 
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
        options={"disp": False, "maxiter": total_iterations_expected, 'ftol': 1e-8, 'gtol': 1e-6}, # disp=False pour éviter la sortie par défaut
        callback=optimization_callback # AJOUT IMPORTANT: Le callback
    )

    # Une fois l'optimisation terminée, assurez-vous de passer à la ligne suivante
    print("\n") 

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
    print(f"  theta: {calibrated_params[2]:.6f}") # corrected 'calibrated' to 'calibrated_params' here
    print(f"  xi: {calibrated_params[3]:.6f}")
    print(f"  rho: {calibrated_params[4]:.6f}")
    print(f"Erreur minimale (Somme des erreurs quadratiques): {min_error:.6f}")

    return {
        "status": "success" if result.success else "failed",
        "calibrated_params": calibrated_params.tolist(),
        "min_error": float(min_error),
        "optimizer_message": result.message,
    }


if __name__ == "__main__":
    ticker = "AAPL"
    
    expirations = get_option_expirations(ticker)
    
    expiration_date_to_calibrate = None
    target_months = 1 
    current_date_dt = datetime.now()

    for exp_str in sorted(expirations):
        exp_dt = datetime.strptime(exp_str, '%Y-%m-%d')
        if (exp_dt - current_date_dt).days >= target_months * 30: 
            expiration_date_to_calibrate = exp_str
            break
    
    if expiration_date_to_calibrate is None and expirations:
        expiration_date_to_calibrate = sorted(expirations)[-1]
        print(f"Warning: No expiration found at least {target_months} month(s) out. Using furthest available: {expiration_date_to_calibrate}")
    elif not expirations:
        print(f"Aucune date d'expiration trouvée pour {ticker}. Impossible de calibrer.")
        exit() 
    
    print(f"Expiration choisie pour la calibration: {expiration_date_to_calibrate}")

    initial_params_heston = (0.25, 0.8180346583803609, 0.25, 0.10166873318442358, -0.5701091463003106) 

    bounds_heston = (
        (0.005, 0.25),    
        (0.1, 8.0),       
        (0.005, 0.25),    
        (0.01, 1.8),      
        (-0.99, -0.1),    
    )

    calibration_result = run_heston_calibration(
        ticker, expiration_date_to_calibrate, initial_params_heston, bounds_heston
    )
    print(calibration_result)