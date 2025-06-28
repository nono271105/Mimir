# mimir/models/heston/heston_model.py

import numpy as np
import cmath # Pour les nombres complexes
from numba import jit, njit, prange # Pour l'accélération avec Numba

@njit
def _heston_char_func_components_numba(phi: float, T: float, r: float, kappa: float, theta: float, sigma: float, rho: float, u_val: float):
    """
    Calcule les composantes C et D de la fonction caractéristique de Heston,
    optimisée pour Numba.
    Cette implémentation est basée sur une formulation robuste couramment utilisée.
    """
    
    # 1. Définition des termes intermédiaires 'a' et 'b'
    # '1j' est l'unité imaginaire en Python
    a_term = u_val * (0 + 1j) * phi - 0.5 * phi**2 
    b_term = kappa - rho * sigma * (0 + 1j) * phi
    
    # 2. Calcul de d_val (le terme clé sous la racine carrée)
    # Correction cruciale du signe et des facteurs.
    d_val = cmath.sqrt(b_term**2 - 2 * a_term * sigma**2)

    # 3. Calcul de g_val (un ratio complexe)
    # Protection contre la division par zéro ou valeurs très petites pour la stabilité
    if abs(b_term + d_val) < 1e-18: # Ajout d'une petite tolérance pour éviter division par zéro
        g_val = 0.0 # Ou gérer l'erreur selon le cas
    else:
        g_val = (b_term - d_val) / (b_term + d_val)
    
    # Termes exponentiels pour C et D
    exp_d_T = cmath.exp(-d_val * T)
    
    # Protection pour le logarithme
    log_arg_denom = (1 - g_val)
    if abs(log_arg_denom) < 1e-18: # Éviter log(0)
        log_term = (0 + 0j) # Gérer comme une valeur nulle si le dénominateur est quasi-nul
    else:
        log_term = cmath.log((1 - g_val * exp_d_T) / log_arg_denom)

    # 4. Calcul de C_val (le premier coefficient de la fonction caractéristique)
    C_val = r * (0 + 1j) * phi * T + (kappa * theta / sigma**2) * (\
              (b_term - d_val) * T - 2 * log_term\
              )
    
    # 5. Calcul de D_val (le second coefficient de la fonction caractéristique)
    D_val_denom = (1 - g_val * exp_d_T)
    if abs(D_val_denom) < 1e-18: # Éviter division par zéro
        D_val = (0 + 0j)
    else:
        D_val = ((b_term - d_val) / sigma**2) * ((1 - exp_d_T) / D_val_denom)
    
    return C_val, D_val

@njit(parallel=True) # Utilisation de prange pour la parallélisation
def _heston_integral_numba(phi_values: np.ndarray, S: float, K: float, T: float, r: float, kappa: float, theta: float, sigma: float, rho: float, v0: float, u_val: float):
    """
    Calcule l'intégrale pour le modèle de Heston en utilisant la règle des trapèzes,
    optimisée avec Numba.
    """
    integral_sum = 0.0 + 0j # Doit être complexe
    
    # Note: L'intégrale va de 0 à un "infini" pratique. 
    # Pour la règle des trapèzes, nous avons besoin des points dphi.
    # Ici, phi_values est un tableau de points d'intégration.
    # Assumons que phi_values est généré avec un pas régulier.
    if len(phi_values) < 2:
        return (0.0 + 0j)

    dphi = phi_values[1] - phi_values[0] # Pas d'intégration

    for i in prange(len(phi_values)):
        phi = phi_values[i]
        
        # Le cas phi=0 est singulier dans l'intégrande (division par phi).
        # On peut gérer cela en prenant la limite ou en l'ignorant pour le premier point
        # et commencer l'intégration à un petit epsilon, ou utiliser une technique de quadrature
        # qui gère les singularités. Pour simplifier ici, ignorons le point phi=0 si c'est le premier.
        if phi == 0:
            continue 

        C, D = _heston_char_func_components_numba(phi, T, r, kappa, theta, sigma, rho, u_val)
        
        f_j_S = cmath.exp(C + D * v0 + (0 + 1j) * phi * cmath.log(S))

        # L'intégrande est exp(-i * phi * log(K)) * f_j_S / (i * phi)
        integrand_term = cmath.exp(-(0 + 1j) * phi * cmath.log(K)) * f_j_S / ((0 + 1j) * phi)
        
        # Pour la règle des trapèzes, on somme (f(x_i) + f(x_{i+1})) / 2 * dx
        # Simplifié ici pour une somme de Riemann pour le moment (plus simple avec prange)
        # Pour une vraie règle des trapèzes, il faudrait ajuster les coefficients aux bords.
        integral_sum += integrand_term 
    
    # La somme doit être multipliée par dphi pour la règle des trapèzes (si c'était une somme de Riemann).
    # Pour des points d'intégration discrétisés, c'est généralement sum(f(x_i) * dx)
    return integral_sum * dphi 


def heston_price(S: float, K: float, T: float, r: float, kappa: float, theta: float, sigma: float, rho: float, v0: float, option_type: str):
    """
    Calcule le prix d'une option européenne vanille en utilisant le modèle de Heston.
    Utilise la formule de Lewis via intégration numérique (Numba-accélérée).
    """
    
    if T <= 0:
        # Valeur intrinsèque si l'option est à maturité ou expirée
        return max(0.0, S - K) if option_type == 'C' else max(0.0, K - S)

    # Plage d'intégration pour phi
    # Une limite supérieure typique est 100-200, mais peut être ajustée.
    # Le nombre de points est important pour la précision.
    num_points = 2000 # Nombre de points d'intégration
    upper_limit = 200 # Limite supérieure de phi
    # Génération des points d'intégration. Nous évitons exactement phi=0 pour l'intégrande.
    phi_values = np.linspace(1e-10, upper_limit, num_points) # Commence à un petit epsilon

    # Calcul de P1
    integral_P1_complex = _heston_integral_numba(phi_values, S, K, T, r, kappa, theta, sigma, rho, v0, u_val=0.5)
    P1 = 0.5 + (integral_P1_complex.real / np.pi) # On prend la partie réelle de l'intégrale

    # Calcul de P2
    integral_P2_complex = _heston_integral_numba(phi_values, S, K, T, r, kappa, theta, sigma, rho, v0, u_val=-0.5)
    P2 = 0.5 + (integral_P2_complex.real / np.pi) # On prend la partie réelle de l'intégrale

    # Calcul du prix de l'option selon la formule de Lewis
    if option_type == 'C':
        price = S * P1 - K * np.exp(-r * T) * P2
    elif option_type == 'P':
        # Formule de parité Put-Call pour Heston: P = C - S + K * exp(-rT)
        # Mais on peut aussi dériver la formule de Put directement de Lewis:
        price = K * np.exp(-r * T) * (1 - P2) - S * (1 - P1)
    else:
        raise ValueError("option_type doit être 'C' ou 'P'.")
        
    return price