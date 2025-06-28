# mimir/models/exotic/pricing.py
from src.models.heston.process import generate_heston_paths
from core.monte_carlo_pricer import run_monte_carlo
from .payoffs import (
    calculate_barrier_payoff,
    calculate_asian_payoff,
    calculate_digital_payoff
)

def price_heston_barrier_option(
    S0: float, V0: float, kappa: float, theta: float, xi: float, rho: float,
    T: float, r: float, N_steps: int, N_simulations: int,
    K: float, barrier_level: float, option_type: str, knock_type: str, in_out: str
) -> float:
    """
    Calcule le prix d'une option barrière sous le modèle de Heston en utilisant Monte Carlo.

    Paramètres:
        Paramètres du modèle Heston: S0, V0, kappa, theta, xi, rho, T, r
        Paramètres de simulation: N_steps, N_simulations
        Paramètres de l'option barrière: K, barrier_level, option_type, knock_type, in_out

    Returns:
        float: Le prix de l'option barrière.
    """
    paths_S, _ = generate_heston_paths(S0, V0, kappa, theta, xi, rho, T, r, N_steps, N_simulations)
    
    # run_monte_carlo a besoin de la fonction de payoff et de ses arguments
    # Les arguments de payoff seront passés à calculate_barrier_payoff
    option_price = run_monte_carlo(
        paths_S, r, T, calculate_barrier_payoff,
        K, barrier_level, option_type, knock_type, in_out
    )
    return option_price

def price_heston_asian_option(
    S0: float, V0: float, kappa: float, theta: float, xi: float, rho: float,
    T: float, r: float, N_steps: int, N_simulations: int,
    K: float, option_type: str
) -> float:
    """
    Calcule le prix d'une option asiatique sous le modèle de Heston en utilisant Monte Carlo.

    Paramètres:
        Paramètres du modèle Heston: S0, V0, kappa, theta, xi, rho, T, r
        Paramètres de simulation: N_steps, N_simulations
        Paramètres de l'option asiatique: K, option_type

    Returns:
        float: Le prix de l'option asiatique.
    """
    paths_S, _ = generate_heston_paths(S0, V0, kappa, theta, xi, rho, T, r, N_steps, N_simulations)
    
    option_price = run_monte_carlo(
        paths_S, r, T, calculate_asian_payoff,
        K, option_type
    )
    return option_price

def price_heston_digital_option(
    S0: float, V0: float, kappa: float, theta: float, xi: float, rho: float,
    T: float, r: float, N_steps: int, N_simulations: int,
    K: float, payoff_amount: float, option_type: str
) -> float:
    """
    Calcule le prix d'une option digitale sous le modèle de Heston en utilisant Monte Carlo.

    Paramètres:
        Paramètres du modèle Heston: S0, V0, kappa, theta, xi, rho, T, r
        Paramètres de simulation: N_steps, N_simulations
        Paramètres de l'option digitale: K, payoff_amount, option_type

    Returns:
        float: Le prix de l'option digitale.
    """
    paths_S, _ = generate_heston_paths(S0, V0, kappa, theta, xi, rho, T, r, N_steps, N_simulations)
    
    option_price = run_monte_carlo(
        paths_S, r, T, calculate_digital_payoff,
        K, payoff_amount, option_type
    )
    return option_price