import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

# --- Demande des paramètres à l'utilisateur ---
try:
    option_type_input = input("Voulez-vous calculer le prix d'une option Call (C) ou Put (P) ? ").upper()
    if option_type_input not in ['C', 'P']:
        raise ValueError("Type d'option invalide. Veuillez entrer 'C' pour Call ou 'P' pour Put.")
    
    S = float(input("Entrez le prix spot actuel (S) : "))
    K = float(input("Entrez le prix d'exercice (K) : "))
    T = float(input("Entrez le temps jusqu'à l'échéance en années (T) : "))
    r = float(input("Entrez le taux d'intérêt sans risque (r, ex: 0.045 pour 4.5%) : "))
    sigma = float(input("Entrez la volatilité (sigma, ex: 0.2 pour 20%) : "))
except ValueError as e:
    print(f"Erreur d'entrée : {e}. Veuillez vous assurer que tous les paramètres sont des nombres et que le type d'option est correct.")
    exit()

# --- Fonction Black-Scholes pour Prix et Grecs ---
def black_scholes_greeks(option_type, S, K, T, r, sigma):
    # Gestion des cas où sigma est très petit ou T est nul pour éviter des erreurs division par zéro ou log de zéro
    if sigma < 1e-9 or T <= 0: # Utilise une petite valeur pour sigma pour éviter les erreurs
        # Retourne 0 pour tous les Grecs si les conditions ne sont pas bonnes
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 
    
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
        price = 0.0 # Ne devrait pas arriver

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
    # Vega est souvent affiché par point de pourcentage de volatilité (divisé par 100)
    vega_per_percent = vega / 100 

    # Theta (en variation par an, souvent converti en variation par jour)
    if option_type == 'C':
        theta = (- (S * n_d1 * sigma) / (2 * sqrt_T) - r * K * np.exp(-r * T) * N_d2)
    elif option_type == 'P':
        theta = (- (S * n_d1 * sigma) / (2 * sqrt_T) + r * K * np.exp(-r * T) * st.norm.cdf(-d2))
    theta_per_day = theta / 365 # Theta par jour

    # Rho (en variation pour 1% de changement du taux sans risque)
    if option_type == 'C':
        rho = K * T * np.exp(-r * T) * N_d2
    elif option_type == 'P':
        rho = -K * T * np.exp(-r * T) * st.norm.cdf(-d2)
    rho_per_percent = rho / 100 # Rho par 1% de changement du taux

    return price, d1, d2, N_d1, N_d2, delta, gamma, vega_per_percent, theta_per_day, rho_per_percent

# --- Calcul du prix de l'option, des paramètres Black-Scholes et des Grecs ---
option_price, d1_val, d2_val, N_d1_val, N_d2_val, delta_val, gamma_val, vega_val, theta_val, rho_val = \
    black_scholes_greeks(option_type_input, S, K, T, r, sigma)

print(f"\n--- Résultats du Modèle Black-Scholes ---")
print(f"Le prix de l'option {option_type_input} est : {option_price:.2f} $")
print(f"d1 = {d1_val:.4f}")
print(f"d2 = {d2_val:.4f}")
print(f"N(d1) = {N_d1_val:.4f}")
print(f"N(d2) = {N_d2_val:.4f}")
print(f"\n--- Les Grecs ---")
print(f"Delta = {delta_val:.4f}")
print(f"Gamma = {gamma_val:.4f}")
print(f"Vega = {vega_val:.4f} (pour 1% de volatilité)")
print(f"Theta = {theta_val:.4f} (par jour)")
print(f"Rho = {rho_val:.4f} (pour 1% de taux d'intérêt)")

# --- Génération des prix du sous-jacent pour le graphique de payoff ---
# Une plage plus large pour bien voir le payoff, centrée autour du strike et du prix spot
# On s'assure que le minimum est >= 0
min_S_range = max(0, min(S, K) * 0.7) 
max_S_range = max(S, K) * 1.3 
S_payoff_range = np.linspace(min_S_range, max_S_range, 200)

# --- Calcul du PROFIT/PERTE NET à l'échéance ---
profit_loss_values = []
for s_at_maturity in S_payoff_range:
    if option_type_input == 'C':
        payoff_brut = max(0, s_at_maturity - K)
    elif option_type_input == 'P':
        payoff_brut = max(0, K - s_at_maturity)
    
    # Le profit/perte net est le payoff brut moins le prix de l'option payé
    profit_loss = payoff_brut - option_price
    profit_loss_values.append(profit_loss)

# --- Tracé du graphique du profit/perte ---
plt.figure(figsize=(10, 6))
plt.plot(S_payoff_range, profit_loss_values, label=f"Profit/Perte de l'option {option_type_input}", color='purple')
plt.axvline(x=K, color='green', linestyle='--', label=f"Prix d'exercice (K = {K})")
plt.axhline(y=0, color='gray', linestyle='-') # Ligne zéro pour le profit/perte

# Marquer le point d'équilibre (breakeven point)
if option_type_input == 'C':
    breakeven_point = K + option_price
elif option_type_input == 'P':
    breakeven_point = K - option_price
    # Assurez-vous que le point d'équilibre n'est pas négatif pour un put
    if breakeven_point < 0: 
        breakeven_point = 0 
        
plt.axvline(x=breakeven_point, color='orange', linestyle=':', label=f"Point d'équilibre ({breakeven_point:.2f})")


plt.title(f"Profit/Perte net de l'option {option_type_input} à l'échéance (Prix de l'option = {option_price:.2f})")
plt.xlabel("Prix du sous-jacent à l'échéance ($)")
plt.ylabel("Profit/Perte ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()