# src/models/bjerksund_stensland_model.py

import numpy as np
import scipy.stats as st  # Pour st.norm.cdf (fonction de distribution normale cumulative)

# --- Fonctions auxiliaires pour le modèle Bjerksund-Stensland ---


def _black_scholes_greeks_internal(
    option_type: str,
    S: float,  # Prix de l'action
    K: float,  # Prix d'exercice
    T: float,  # Temps jusqu'à l'échéance
    r: float,  # Taux sans risque
    sigma: float,  # Volatilité
    q: float = 0.0,  # Rendement des dividendes continu
):
    """
    Calcule le prix d'une option européenne et ses Grecs selon Black-Scholes-Merton.
    Cette fonction est interne et adaptée aux besoins du modèle Bjerksund-Stensland.
    """
    if T <= 0 or sigma <= 1e-10:
        # Retourne la valeur intrinsèque si T=0 ou volatilité négligeable
        price = max(0.0, S - K) if option_type == "C" else max(0.0, K - S)
        # Retourne également des valeurs pour d1, d2, N_d1, N_d2, delta, gamma, vega, theta, rho
        # pour maintenir la signature de retour, bien qu'elles soient nulles ou non pertinentes ici.
        return price, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Assurez-vous que sqrt(T) n'est pas zéro ou proche de zéro si T est très petit mais positif.
    sqrt_T = np.sqrt(T)
    if sqrt_T < 1e-10:  # Gérer le cas T très proche de zéro
        price = max(0.0, S - K) if option_type == "C" else max(0.0, K - S)
        return price, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    N_d1 = st.norm.cdf(d1)
    N_d2 = st.norm.cdf(d2)
    n_d1 = st.norm.pdf(d1)  # Densité de probabilité pour Gamma et Vega

    if option_type == "C":
        price = S * np.exp(-q * T) * N_d1 - K * np.exp(-r * T) * N_d2
        delta = np.exp(-q * T) * N_d1
        theta = (
            -(S * np.exp(-q * T) * n_d1 * sigma) / (2 * sqrt_T)
            - r * K * np.exp(-r * T) * N_d2
            + q * S * np.exp(-q * T) * N_d1
        )
        rho = K * T * np.exp(-r * T) * N_d2
    elif option_type == "P":
        price = K * np.exp(-r * T) * st.norm.cdf(-d2) - S * np.exp(
            -q * T
        ) * st.norm.cdf(-d1)
        delta = np.exp(-q * T) * (N_d1 - 1)
        theta = (
            -(S * np.exp(-q * T) * n_d1 * sigma) / (2 * sqrt_T)
            + r * K * np.exp(-r * T) * st.norm.cdf(-d2)
            - q * S * np.exp(-q * T) * st.norm.cdf(-d1)
        )
        rho = -K * T * np.exp(-r * T) * st.norm.cdf(-d2)
    else:
        raise ValueError(
            "Type d'option invalide. Utilisez 'C' pour Call ou 'P' pour Put."
        )

    gamma = np.exp(-q * T) * n_d1 / (S * sigma * sqrt_T)
    vega = S * np.exp(-q * T) * n_d1 * sqrt_T

    return price, d1, d2, N_d1, N_d2, delta, gamma, vega, theta, rho


def newton_raphson_bs_american_call(
    S_current: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    epsilon: float = 1e-6,
) -> float:
    """
    Résout la frontière d'exercice anticipé optimale S* pour une option Call américaine
    en utilisant la méthode de Newton-Raphson.
    """
    # Meilleure estimation initiale. Choisir une valeur raisonnable légèrement supérieure à K.
    # Pour la robustesse, S_current * 1.05 ou K * 1.1 sont de bons points de départ.
    # S_star doit être > K pour un Call américain.
    S_star_old = max(K * 1.1, S_current + 0.1)

    # Assurez-vous que S_star_old n'est pas zéro pour éviter log(0)
    if S_star_old <= 0:
        S_star_old = K + 0.01  # Fallback sûr

    for _ in range(200):  # Augmenter les itérations pour une meilleure convergence
        # Calcule le prix Black-Scholes et le Delta à S_star_old (en utilisant S_star_old comme prix spot)
        euro_call_at_S_star, _, _, _, _, delta, _, _, _, _ = (
            _black_scholes_greeks_internal("C", S_star_old, K, T, r, sigma, q)
        )

        # La fonction F(S*) dont on cherche la racine
        # F(S*) = IntrinsicValue(S*) - EuropeanCallPrice(S*)
        # F(S*) = (S* - K) - C_euro(S*)
        F_S_star = (S_star_old - K) - euro_call_at_S_star

        # La dérivée de F(S*) par rapport à S* est d(S* - K)/dS* - d(C_euro)/dS* = 1 - Delta_euro(S*)
        F_prime_S_star = 1 - delta

        if abs(F_prime_S_star) < 1e-10:  # Évite la division par zéro
            # Si la dérivée est proche de zéro, faire un petit pas fixe pour tenter de sortir de la zone plate
            S_star_new = S_star_old + 0.001 * K
        else:
            S_star_new = S_star_old - F_S_star / F_prime_S_star

        if abs(S_star_new - S_star_old) < epsilon:  # Critère de convergence
            return S_star_new

        S_star_old = S_star_new

    return S_star_old  # Retourne la meilleure approximation après le nombre max d'itérations


def newton_raphson_bs_american_put(
    S_current: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    epsilon: float = 1e-6,
) -> float:
    """
    Résout la frontière d'exercice anticipé optimale S* pour une option Put américaine
    en utilisant la méthode de Newton-Raphson.
    """
    # Meilleure estimation initiale. Choisir une valeur raisonnable légèrement inférieure à K.
    # S_star doit être < K pour un Put américain.
    S_star_old = min(K * 0.9, S_current - 0.1)

    # Assurez-vous que S_star_old n'est pas zéro ou négatif.
    if S_star_old <= 0:
        S_star_old = K - 0.01  # Fallback sûr, doit être positif

    for _ in range(200):  # Augmenter les itérations
        euro_put_at_S_star, _, _, _, _, delta, _, _, _, _ = (
            _black_scholes_greeks_internal("P", S_star_old, K, T, r, sigma, q)
        )

        # F(S*) = IntrinsicValue(S*) - EuropeanPutPrice(S*)
        # F(S*) = (K - S*) - P_euro(S*)
        F_S_star = (K - S_star_old) - euro_put_at_S_star

        # F'(S*) = d(K - S*)/dS* - d(P_euro)/dS* = -1 - Delta_euro(S*)
        F_prime_S_star = -1 - delta

        if abs(F_prime_S_star) < 1e-10:
            S_star_new = (
                S_star_old - 0.001 * K
            )  # Petit pas si la dérivée est proche de zéro
        else:
            S_star_new = S_star_old - F_S_star / F_prime_S_star

        if abs(S_star_new - S_star_old) < epsilon:
            return S_star_new

        S_star_old = S_star_new

    return S_star_old


def bjerksund_stensland_2002(
    option_type: str,
    S: float,  # Prix actuel du sous-jacent
    K: float,  # Prix d'exercice
    T: float,  # Temps jusqu'à l'échéance en années
    r: float,  # Taux sans risque
    sigma: float,  # Volatilité
    q: float = 0.0,  # Rendement des dividendes continu
) -> float:
    """
    Calcule le prix d'une option américaine (Call ou Put) en utilisant
    l'approximation de Bjerksund-Stensland (2002).

    Référence: Bjerksund, P., & Stensland, G. (2002). "Closed-form approximation of American options."
    """

    # Gestion des cas limites (échéance passée ou volatilité nulle)
    if T <= 1e-10 or sigma <= 1e-10:  # Utiliser une petite valeur pour la robustesse
        return max(0.0, S - K) if option_type == "C" else max(0.0, K - S)

    # Coût de portage
    b = r - q

    # Calcul de beta_val (lambda dans la notation originale B-S 2002)
    # Assurez-vous que sigma**2 n'est pas zéro. Géré par la condition sigma <= 1e-10 ci-dessus.
    sigma_squared = sigma**2
    beta_val = (0.5 - b / sigma_squared) + np.sqrt(
        ((b / sigma_squared) - 0.5) ** 2 + 2 * r / sigma_squared
    )

    # Calcul du prix de l'option européenne correspondante
    euro_price = _black_scholes_greeks_internal(option_type, S, K, T, r, sigma, q)[0]

    if option_type == "C":
        # Pour les Calls, si b >= r (rendement de dividende faible/nul), l'exercice anticipé n'est pas optimal
        # L'option américaine Call se comporte comme une option européenne Call.
        if b >= r:
            return euro_price

        # Calcul de la frontière d'exercice anticipé (I)
        I = newton_raphson_bs_american_call(S, K, T, r, sigma, q)

        # Si le prix actuel est déjà à la frontière ou au-delà, l'option est exercée immédiatement
        if S >= I:
            return S - K

        # Calcul des termes y1 et y2 (variables intermédiaires spécifiques à B-S 2002)
        sqrt_T = np.sqrt(T)
        y1 = (np.log(S / I) + (b - 0.5 * sigma_squared) * T) / (sigma * sqrt_T)
        y2 = y1 - sigma * sqrt_T

        # Terme additionnel pour l'exercice anticipé (A2 dans la notation du papier)
        A2_term = K * np.exp(-r * T) * st.norm.cdf(-y2) - S * np.exp(
            -q * T
        ) * st.norm.cdf(-y1)

        # Prix final de l'option Call américaine
        price = euro_price + A2_term * (S / I) ** beta_val
        return price

    elif option_type == "P":
        # Calcul de la frontière d'exercice anticipé (I)
        I = newton_raphson_bs_american_put(S, K, T, r, sigma, q)

        # Si le prix actuel est déjà à la frontière ou en dessous, l'option est exercée immédiatement
        if S <= I:
            return K - S

        # Calcul des termes y1 et y2 (variables intermédiaires spécifiques à B-S 2002)
        sqrt_T = np.sqrt(T)
        y1 = (np.log(S / I) + (b - 0.5 * sigma_squared) * T) / (sigma * sqrt_T)
        y2 = y1 - sigma * sqrt_T

        A2_prime_term = K * np.exp(-r * T) * st.norm.cdf(-y2) - S * np.exp(
            -q * T
        ) * st.norm.cdf(-y1)

        price = euro_price + A2_prime_term * (I / S) ** beta_val
        return price

    else:
        raise ValueError(
            "Type d'option invalide. Utilisez 'C' pour Call ou 'P' pour Put."
        )
