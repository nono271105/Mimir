# mimir/src/models/european/pricing.py

import numpy as np
import cmath  # Pour les nombres complexes
from scipy.integrate import quad  # Pour l'intégration numérique

# Pas besoin de calculate_T_from_expiration ici, il est géré en amont.

# Function characteristic of the Heston model (used for P1 and P2)
def heston_char_function(phi, kappa, theta, xi, rho, V0, r, T, S0, u_param):
    alpha = u_param

    # Defensive check for xi and T
    # If xi or T are too small, the model can become degenerate or numerically unstable.
    if xi <= 1e-6 or T <= 1e-6:
        return 0.0 + 0.0j # Return complex zero to penalize problematic parameters

    # Terme complexe discriminant (gamma)
    term_inside_sqrt = (kappa - rho * xi * 1j * phi)**2 + xi**2 * (1j * phi * alpha + 1j * phi**2)
    
    # Ensure term_inside_sqrt is not extremely small or problematic before sqrt
    # Add a small complex epsilon to prevent sqrt(0) or near-zero issues
    if np.abs(term_inside_sqrt) < 1e-15:
        gamma = cmath.sqrt(1e-15 + 0j) 
    else:
        gamma = cmath.sqrt(term_inside_sqrt) 

    # Terme D(phi)
    num_D = kappa - rho * xi * 1j * phi - gamma

    denom_G_phi = kappa - rho * xi * 1j * phi + gamma
    # Handle near-zero denominator for G_phi
    if np.abs(denom_G_phi) < 1e-18:
        G_phi = (num_D / 1e-18) 
    else:
        G_phi = num_D / denom_G_phi
    
    # Dénominateur principal pour B (term_denom_B)
    exp_gamma_T = cmath.exp(-gamma * T) # Use cmath.exp for complex numbers
    term_denom_B = (1 - G_phi * exp_gamma_T)
    
    # Handle near-zero term_denom_B
    if np.abs(term_denom_B) < 1e-18:
        term_denom_B = 1e-18 + 0j # Ensure it's complex if pushing

    # Ensure xi is not too small before division
    if xi**2 < 1e-18:
        return 0.0 + 0.0j # Avoid division by near zero xi^2

    B = num_D / (xi**2 * term_denom_B) * (1 - exp_gamma_T)

    # Terme A(phi)
    denom_log_internal = (1 - G_phi)
    # Handle near-zero denom_log_internal
    if np.abs(denom_log_internal) < 1e-18:
        denom_log_internal = 1e-18 + 0j

    # Ensure argument to cmath.log is not problematic
    log_arg = term_denom_B / denom_log_internal
    if np.abs(log_arg) < 1e-18: # Avoid log(0)
        log_term_A = cmath.log(1e-18) 
    else:
        log_term_A = cmath.log(log_arg)
    
    A = 1j * phi * r * T + (kappa * theta / xi**2) * (num_D * T - 2 * log_term_A)
    
    # The final characteristic function - ensure finite value before returning
    result = cmath.exp(A + B * V0) # Use cmath.exp for complex numbers
    
    if not np.isfinite(result): # If result is inf or NaN
        return 0.0 + 0.0j # Return a complex zero to penalize

    return result


# Integrand for P1 (for a call)
def integrand_P1(phi, kappa, theta, xi, rho, V0, r, T, S0, K):
    char_func_val = heston_char_function(phi, kappa, theta, xi, rho, V0, r, T, S0, u_param=0.5)
    
    # If char_func_val is not finite (due to previous checks), return 0
    if not np.isfinite(char_func_val):
        return 0.0
    
    if phi == 0: # Avoid division by zero at phi=0
        return 0.0 
        
    # Ensure (1j * phi) is not near zero
    if np.abs(phi) < 1e-18:
        return 0.0
        
    term = (cmath.exp(-1j * phi * cmath.log(K)) * char_func_val / (1j * phi)) # Use cmath for complex log
    
    if not np.isfinite(term): # If term itself becomes inf or NaN
        return 0.0
        
    return term.real

# Integrand for P2 (for a call)
def integrand_P2(phi, kappa, theta, xi, rho, V0, r, T, S0, K):
    char_func_val = heston_char_function(phi, kappa, theta, xi, rho, V0, r, T, S0, u_param=-0.5)

    if not np.isfinite(char_func_val):
        return 0.0

    if phi == 0:
        return 0.0
        
    if np.abs(phi) < 1e-18:
        return 0.0

    term = (cmath.exp(-1j * phi * cmath.log(K)) * char_func_val / (1j * phi)) # Use cmath for complex log

    if not np.isfinite(term):
        return 0.0

    return term.real


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
    
    # Handle extremely short maturities (increased threshold to 7 days for more robustness)
    if T <= 7/365.0: 
        if option_type == 'C':
            return max(0, S0 - K)
        elif option_type == 'P':
            return max(0, K - S0)
    
    # Bornes d'intégration initiales, peuvent être ajustées
    integration_limit = 200 # Toujours une bonne valeur de départ

    # Options pour