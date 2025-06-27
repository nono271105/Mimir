import numpy as np
import scipy.stats as st

def black_scholes_greeks(option_type: str, S: float, K: float, T: float, r: float, sigma: float, dividend_yield: float = 0.0):
    """
    Calcule le prix d'une option européenne (Call ou Put) et ses Grecs
    selon le modèle de Black-Scholes-Merton, incluant la gestion des dividendes continus.

    Args:
        option_type (str): 'C' pour Call, 'P' pour Put.
        S (float): Prix spot actuel du sous-jacent.
        K (float): Prix d'exercice de l'option.
        T (float): Temps jusqu'à l'échéance en années.
        r (float): Taux d'intérêt sans risque annuel (ex: 0.045 pour 4.5%).
        sigma (float): Volatilité annuelle du sous-jacent (ex: 0.2 pour 20%).
        dividend_yield (float): Rendement annuel des dividendes (ex: 0.01 pour 1%). Par défaut à 0.0.

    Returns:
        tuple: (price, d1, d2, N(d1), N(d2), delta, gamma, vega_per_percent, theta_per_day, rho_per_percent)
               Retourne 0 pour tous les Grecs et le prix si T <= 0 ou sigma est trop petit.
    """
    # Gestion des cas limites pour éviter les erreurs de division par zéro ou log de zéro
    if T <= 0:
        # À l'échéance (T=0), le prix est le payoff intrinsèque
        if option_type == 'C':
            price = max(0, S - K)
        elif option_type == 'P':
            price = max(0, K - S)
        else:
            price = 0.0 # Cas non valide

        # Grecs à l'échéance (ou non définis pour T=0)
        # Pour T=0, les dividendes n'ont pas d'impact sur le prix ou les Grecs
        delta = 1.0 if option_type == 'C' and S > K else (0.0 if option_type == 'C' else (-1.0 if S < K else 0.0))
        gamma = 0.0 # Delta devient binaire à l'échéance
        vega_per_percent = 0.0
        theta_per_day = 0.0
        rho_per_percent = 0.0
        d1 = d2 = N_d1 = N_d2 = 0.0 # Non définis ou sans signification à T=0
        return price, d1, d2, N_d1, N_d2, delta, gamma, vega_per_percent, theta_per_day, rho_per_percent

    if sigma < 1e-9: # Volatilité quasi nulle, comportement similaire au cas déterministe
        # Avec dividendes, le prix forward est S * exp((r - q) * T)
        S_forward = S * np.exp((r - dividend_yield) * T)
        if option_type == 'C':
            price = max(0, S_forward - K) * np.exp(-r * T) # Valeur intrinsèque actualisée
        elif option_type == 'P':
            price = max(0, K - S_forward) * np.exp(-r * T)
        else:
            price = 0.0

        # Les Grecs pour sigma -> 0 sont également dégénérés
        delta = 1.0 if option_type == 'C' and S_forward > K else (0.0 if option_type == 'C' else (-1.0 if S_forward < K else 0.0))
        gamma = 0.0
        vega_per_percent = 0.0
        theta_per_day = 0.0
        rho_per_percent = 0.0
        d1 = d2 = N_d1 = N_d2 = 0.0
        return price, d1, d2, N_d1, N_d2, delta, gamma, vega_per_percent, theta_per_day, rho_per_percent


    sqrt_T = np.sqrt(T)
    
    # Formules d1 et d2 ajustées pour le rendement des dividendes (q)
    d1 = (np.log(S / K) + (r - dividend_yield + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    N_d1 = st.norm.cdf(d1)
    N_d2 = st.norm.cdf(d2)
    n_d1 = st.norm.pdf(d1) # Densité de probabilité normale standard de d1

    # Prix de l'option ajusté pour le rendement des dividendes (q)
    if option_type == 'C':
        price = S * np.exp(-dividend_yield * T) * N_d1 - K * np.exp(-r * T) * N_d2
    elif option_type == 'P':
        price = K * np.exp(-r * T) * st.norm.cdf(-d2) - S * np.exp(-dividend_yield * T) * st.norm.cdf(-d1)
    else:
        price = 0.0 # Ne devrait pas arriver avec la validation d'entrée

    # Grecs ajustés pour le rendement des dividendes (q)
    # Delta
    if option_type == 'C':
        delta = np.exp(-dividend_yield * T) * N_d1
    elif option_type == 'P':
        delta = np.exp(-dividend_yield * T) * (N_d1 - 1)

    # Gamma
    gamma = np.exp(-dividend_yield * T) * n_d1 / (S * sigma * sqrt_T)

    # Vega
    vega = S * np.exp(-dividend_yield * T) * n_d1 * sqrt_T
    vega_per_percent = vega / 100 

    # Theta (en variation par an, souvent converti en variation par jour)
    if option_type == 'C':
        theta = (- (S * np.exp(-dividend_yield * T) * n_d1 * sigma) / (2 * sqrt_T)
                 - r * K * np.exp(-r * T) * N_d2
                 + dividend_yield * S * np.exp(-dividend_yield * T) * N_d1) # Terme de dividende pour Theta Call
    elif option_type == 'P':
        theta = (- (S * np.exp(-dividend_yield * T) * n_d1 * sigma) / (2 * sqrt_T)
                 + r * K * np.exp(-r * T) * st.norm.cdf(-d2)
                 - dividend_yield * S * np.exp(-dividend_yield * T) * st.norm.cdf(-d1)) # Terme de dividende pour Theta Put
    theta_per_day = theta / 365 

    # Rho (en variation pour 1% de changement du taux sans risque)
    if option_type == 'C':
        rho = K * T * np.exp(-r * T) * N_d2
    elif option_type == 'P':
        rho = -K * T * np.exp(-r * T) * st.norm.cdf(-d2)
    rho_per_percent = rho / 100 

    return price, d1, d2, N_d1, N_d2, delta, gamma, vega_per_percent, theta_per_day, rho_per_percent