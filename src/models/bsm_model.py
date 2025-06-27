import numpy as np
import scipy.stats as st

def black_scholes_greeks(option_type: str, S: float, K: float, T: float, r: float, sigma: float):
    """
    Calcule le prix d'une option européenne (Call ou Put) et ses Grecs
    selon le modèle de Black-Scholes-Merton.

    Args:
        option_type (str): 'C' pour Call, 'P' pour Put.
        S (float): Prix spot actuel du sous-jacent.
        K (float): Prix d'exercice de l'option.
        T (float): Temps jusqu'à l'échéance en années.
        r (float): Taux d'intérêt sans risque annuel (ex: 0.045 pour 4.5%).
        sigma (float): Volatilité annuelle du sous-jacent (ex: 0.2 pour 20%).

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
        delta = 1.0 if option_type == 'C' and S > K else (0.0 if option_type == 'C' else (-1.0 if S < K else 0.0))
        gamma = 0.0 # Delta devient binaire à l'échéance
        vega_per_percent = 0.0
        theta_per_day = 0.0
        rho_per_percent = 0.0
        d1 = d2 = N_d1 = N_d2 = 0.0 # Non définis ou sans signification à T=0
        return price, d1, d2, N_d1, N_d2, delta, gamma, vega_per_percent, theta_per_day, rho_per_percent

    if sigma < 1e-9: # Volatilité quasi nulle, comportement similaire au cas déterministe
        if option_type == 'C':
            price = max(0, S * np.exp(r * T) - K) * np.exp(-r * T) # Valeur intrinsèque actualisée
        elif option_type == 'P':
            price = max(0, K - S * np.exp(r * T)) * np.exp(-r * T)
        else:
            price = 0.0

        # Les Grecs pour sigma -> 0 sont également dégénérés
        delta = 1.0 if option_type == 'C' and S * np.exp(r * T) > K else (0.0 if option_type == 'C' else (-1.0 if S * np.exp(r * T) < K else 0.0))
        gamma = 0.0
        vega_per_percent = 0.0
        theta_per_day = 0.0
        rho_per_percent = 0.0
        d1 = d2 = N_d1 = N_d2 = 0.0
        return price, d1, d2, N_d1, N_d2, delta, gamma, vega_per_percent, theta_per_day, rho_per_percent


    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    N_d1 = st.norm.cdf(d1)
    N_d2 = st.norm.cdf(d2)
    n_d1 = st.norm.pdf(d1) # Densité de probabilité normale standard de d1

    # Prix de l'option
    if option_type == 'C':
        price = S * N_d1 - K * np.exp(-r * T) * N_d2
    elif option_type == 'P':
        price = K * np.exp(-r * T) * st.norm.cdf(-d2) - S * st.norm.cdf(-d1)
    else:
        price = 0.0 # Ne devrait pas arriver avec la validation d'entrée

    # Grecs
    # Delta
    if option_type == 'C':
        delta = N_d1
    elif option_type == 'P':
        delta = N_d1 - 1

    # Gamma
    gamma = n_d1 / (S * sigma * sqrt_T)

    # Vega
    vega = S * n_d1 * sqrt_T
    vega_per_percent = vega / 100 

    # Theta (en variation par an, souvent converti en variation par jour)
    if option_type == 'C':
        theta = (- (S * n_d1 * sigma) / (2 * sqrt_T) - r * K * np.exp(-r * T) * N_d2)
    elif option_type == 'P':
        theta = (- (S * n_d1 * sigma) / (2 * sqrt_T) + r * K * np.exp(-r * T) * st.norm.cdf(-d2))
    theta_per_day = theta / 365 

    # Rho (en variation pour 1% de changement du taux sans risque)
    if option_type == 'C':
        rho = K * T * np.exp(-r * T) * N_d2
    elif option_type == 'P':
        rho = -K * T * np.exp(-r * T) * st.norm.cdf(-d2)
    rho_per_percent = rho / 100 

    return price, d1, d2, N_d1, N_d2, delta, gamma, vega_per_percent, theta_per_day, rho_per_percent