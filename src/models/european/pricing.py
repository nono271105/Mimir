# models/european/pricing.py
import numpy as np
import cmath # Pour les nombres complexes
from scipy.integrate import quad # Pour l'intégration numérique

# Fonction de pricing analytique pour Heston
def heston_char_function(phi, kappa, theta, xi, rho, V0, r, T, S0):
    """
    Fonction caractéristique du modèle de Heston.
    """
    # Paramètres intermédiaires pour simplifier
    a = kappa * theta
    u = -0.5
    b = kappa

    # Cas pour les options Call (u=0.5) et Put (u=-0.5)
    # Ici, nous utilisons les formules pour le log-prix, donc u = -0.5
    # Pour le calcul de P1 et P2 dans la formule de Heston, u et b changent.
    # On va définir P1 et P2 séparément pour plus de clarté.

    # D pour P1 et P2
    d1 = np.sqrt((rho * xi * phi * 1j - b)**2 - xi**2 * (2 * u * phi * 1j - phi**2))
    d2 = np.sqrt((rho * xi * phi * 1j - b)**2 - xi**2 * (2 * (u + 1) * phi * 1j - phi**2))

    # g pour P1 et P2
    g1 = (b - rho * xi * phi * 1j + d1) / (b - rho * xi * phi * 1j - d1)
    g2 = (b - rho * xi * phi * 1j + d2) / (b - rho * xi * phi * 1j - d2)

    # C pour P1 et P2
    C1 = r * phi * 1j * T + (a / xi**2) * ((b - rho * xi * phi * 1j + d1) * T - 2 * np.log((1 - g1 * np.exp(d1 * T)) / (1 - g1)))
    C2 = r * phi * 1j * T + (a / xi**2) * ((b - rho * xi * phi * 1j + d2) * T - 2 * np.log((1 - g2 * np.exp(d2 * T)) / (1 - g2)))

    # D pour P1 et P2
    D1 = (b - rho * xi * phi * 1j + d1) / xi**2 * ((1 - np.exp(d1 * T)) / (1 - g1 * np.exp(d1 * T)))
    D2 = (b - rho * xi * phi * 1j + d2) / xi**2 * ((1 - np.exp(d2 * T)) / (1 - g2 * np.exp(d2 * T)))

    return np.exp(C1 + D1 * V0 + 1j * phi * np.log(S0)), np.exp(C2 + D2 * V0 + 1j * phi * np.log(S0))


def integrand_P1(phi, kappa, theta, xi, rho, V0, r, T, S0, K):
    """Intégrand pour P1 dans la formule de Heston."""
    char_func_val_P1, _ = heston_char_function(phi, kappa, theta, xi, rho, V0, r, T, S0)
    return (np.exp(-1j * phi * np.log(K)) * char_func_val_P1).real / (1j * phi)

def integrand_P2(phi, kappa, theta, xi, rho, V0, r, T, S0, K):
    """Intégrand pour P2 dans la formule de Heston."""
    _, char_func_val_P2 = heston_char_function(phi, kappa, theta, xi, rho, V0, r, T, S0)
    return (np.exp(-1j * phi * np.log(K)) * char_func_val_P2).real / (1j * phi)


def price_heston_european_option(
    S0: float,
    V0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    T: float,
    r: float,
    K: float,
    option_type: str,
    # N_steps et N_simulations ne sont plus nécessaires pour l'analytique
    # N_steps: int = 252,
    # N_simulations: int = 100000,
) -> float:
    """
    Calcule le prix d'une option européenne sous le modèle de Heston via la formule analytique.
    """
    # Les lignes de débogage ne sont plus nécessaires, mais vous pouvez les laisser si vous voulez
    # print(f"DEBUG (pricing_func): Inside price_heston_european_option. File: {__file__}")
    # print(f"DEBUG (pricing_func): Received r = {r}")

    # Calcul de P1
    integral_P1, _ = quad(
        integrand_P1,
        0,
        np.inf, # Limite d'intégration jusqu'à l'infini
        args=(kappa, theta, xi, rho, V0, r, T, S0, K),
        limit=1000 # Augmenter le nombre de points pour l'intégration
    )
    P1 = 0.5 + (1 / np.pi) * integral_P1

    # Calcul de P2
    integral_P2, _ = quad(
        integrand_P2,
        0,
        np.inf,
        args=(kappa, theta, xi, rho, V0, r, T, S0, K),
        limit=1000
    )
    P2 = 0.5 + (1 / np.pi) * integral_P2

    if option_type == "C":
        option_price = S0 * P1 - K * np.exp(-r * T) * P2
    elif option_type == "P":
        # Formule de parité Put-Call pour les options européennes
        # P = C - S0 + K * exp(-rT)
        call_price = S0 * P1 - K * np.exp(-r * T) * P2
        option_price = call_price - S0 + K * np.exp(-r * T)
    else:
        raise ValueError("option_type doit être 'C' (Call) ou 'P' (Put).")

    return option_price