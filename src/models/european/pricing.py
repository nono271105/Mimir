# mimir/src/models/european/pricing.py
import numpy as np
import cmath # Make sure this import is present at the top
from scipy.integrate import quad

# ... (rest of your imports and calculate_T_from_expiration function) ...


# Function characteristic of the Heston model (used for P1 and P2)
def heston_char_function(phi, kappa, theta, xi, rho, V0, r, T, S0, u_param):
    alpha = u_param

    # Terme complexe discriminant (gamma)
    # Protection contre les arguments de sqrt qui rendraient gamma numériquement instable
    term_inside_sqrt = (kappa - rho * xi * 1j * phi)**2 + xi**2 * (1j * phi * alpha + 1j * phi**2)

    # Use cmath.sqrt for complex square roots for better numerical stability
    gamma = cmath.sqrt(term_inside_sqrt) 

    # Terme D(phi) - c'est le 'B' de certaines notations
    num_D = kappa - rho * xi * 1j * phi - gamma

    # Dénominateur de la fraction interne dans A et B (appelons-le G_phi)
    denom_G_phi = kappa - rho * xi * 1j * phi + gamma
    epsilon_denom_G = 1e-18 # Small value to avoid division by zero
    if np.abs(denom_G_phi) < epsilon_denom_G:
        G_phi = (num_D / epsilon_denom_G) # Fallback if denominator is near zero
    else:
        G_phi = num_D / denom_G_phi
    
    # Dénominateur principal pour B (term_denom_B)
    term_denom_B = (1 - G_phi * np.exp(-gamma * T))
    epsilon_denom_B = 1e-18 # Small value for stability
    if np.abs(term_denom_B) < epsilon_denom_B:
        term_denom_B = epsilon_denom_B 

    B = num_D / (xi**2 * term_denom_B) * (1 - np.exp(-gamma * T))

    # Terme A(phi)
    denom_log_internal = (1 - G_phi)
    epsilon_log_internal = 1e-18
    if np.abs(denom_log_internal) < epsilon_log_internal:
        log_term_A = np.log(term_denom_B / epsilon_log_internal) # Handle near-zero case
    else:
        log_term_A = np.log(term_denom_B / denom_log_internal)
    
    A = 1j * phi * r * T + (kappa * theta / xi**2) * (num_D * T - 2 * log_term_A)
    
    # The final characteristic function
    return np.exp(A + B * V0)


# Integrand for P1 (for a call)
def integrand_P1(phi, kappa, theta, xi, rho, V0, r, T, S0, K):
    char_func_val = heston_char_function(phi, kappa, theta, xi, rho, V0, r, T, S0, u_param=0.5)
    if phi == 0:
        # Limit as phi -> 0. For P1, it's (S0/K) * exp(-rT). For P2, it's 1.
        # This is a critical point for numerical stability.
        # More correctly, the limit of (exp(-1j * phi * ln K) * char_func / (1j * phi))
        # as phi -> 0 is -i * ln(K) * char_func(0) * (1/ (i * 1)) = -ln(K) * char_func(0)
        # However, the quad function usually handles this limit itself.
        # For a smooth integrand at 0, 0.0 is often the correct contribution
        return 0.0 
    return (np.exp(-1j * phi * np.log(K)) * char_func_val / (1j * phi)).real

# Integrand for P2 (for a call)
def integrand_P2(phi, kappa, theta, xi, rho, V0, r, T, S0, K):
    char_func_val = heston_char_function(phi, kappa, theta, xi, rho, V0, r, T, S0, u_param=-0.5)
    if phi == 0:
        return 0.0
    return (np.exp(-1j * phi * np.log(K)) * char_func_val / (1j * phi)).real


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
) -> float:
    
    # Handle extremely short maturities
    if T <= 1/365.0: # If T is 1 day or less
        # Use Black-Scholes for very short maturities or intrinsic value.
        # For simplicity, for T=0, it's intrinsic value. For very small T, it's close.
        # This prevents integration issues for T near zero.
        if option_type == 'C':
            return max(0, S0 - K)
        elif option_type == 'P':
            return max(0, K - S0)
    
    # Bornes d'intégration initiales, peuvent être ajustées
    integration_limit = 200 # Aumenté pour plus de précision

    # Options pour quad - augmenter les limites et réduire les tolérances pour plus de précision
    quad_options = {
        'limit': 5000,  # Augmenter le nombre max de sous-intervalles
        'epsabs': 1e-12, # Tolérance absolue plus stricte
        'epsrel': 1e-12  # Tolérance relative plus stricte
    }

    # Calcul de P1
    try:
        integral_P1, _ = quad(
            integrand_P1,
            0,
            integration_limit, 
            args=(kappa, theta, xi, rho, V0, r, T, S0, K),
            **quad_options
        )
        P1 = 0.5 + (1 / np.pi) * integral_P1
    except Exception as e:
        # If integration fails, return a value that strongly penalizes (or makes it clear it failed)
        # print(f"DEBUG: P1 integration failed for K={K}, T={T}. Error: {e}")
        return 0.0 # Returning 0.0 will cause high error and push optimizer away

    # Calcul de P2
    try:
        integral_P2, _ = quad(
            integrand_P2,
            0,
            integration_limit,
            args=(kappa, theta, xi, rho, V0, r, T, S0, K),
            **quad_options
        )
        P2 = 0.5 + (1 / np.pi) * integral_P2
    except Exception as e:
        # print(f"DEBUG: P2 integration failed for K={K}, T={T}. Error: {e}")
        return 0.0 # Returning 0.0 will cause high error and push optimizer away


    # Formule de pricing pour Call et Put
    if option_type == "C":
        option_price = S0 * P1 - K * np.exp(-r * T) * P2
    elif option_type == "P":
        # Put-Call Parity
        call_price = S0 * P1 - K * np.exp(-r * T) * P2
        option_price = call_price - S0 + K * np.exp(-r * T)
    else:
        raise ValueError("option_type doit être 'C' (Call) ou 'P' (Put).")

    return max(0, option_price)