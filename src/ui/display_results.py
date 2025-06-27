import matplotlib.pyplot as plt
import numpy as np


def display_bsm_results(
    option_type: str,
    option_price: float,
    d1: float,
    d2: float,
    N_d1: float,
    N_d2: float,
    delta: float,
    gamma: float,
    vega: float,
    theta: float,
    rho: float,
):
    """
    Affiche les résultats du calcul de l'option (prix, d-values, N-values, Grecs).

    Args:
        option_type (str): Type de l'option ('C' ou 'P').
        option_price (float): Prix calculé de l'option.
        d1 (float): Valeur de d1 du modèle Black-Scholes.
        d2 (float): Valeur de d2 du modèle Black-Scholes.
        N_d1 (float): N(d1), probabilité associée à d1.
        N_d2 (float): N(d2), probabilité associée à d2.
        delta (float): Valeur du Delta.
        gamma (float): Valeur du Gamma.
        vega (float): Valeur du Vega (pour 1% de volatilité).
        theta (float): Valeur du Theta (par jour).
        rho (float): Valeur du Rho (pour 1% de taux d'intérêt).
    """
    print(f"\n--- Résultats du Modèle Black-Scholes ---")
    print(f"Le prix de l'option {option_type} est : {option_price:.2f} $")
    print(f"d1 = {d1:.4f}")
    print(f"d2 = {d2:.4f}")
    print(f"N(d1) = {N_d1:.4f}")
    print(f"N(d2) = {N_d2:.4f}")
    print(f"\n--- Les Grecs ---")
    print(f"Delta = {delta:.4f}")
    print(f"Gamma = {gamma:.4f}")
    print(f"Vega = {vega:.4f} (pour 1% de volatilité)")
    print(f"Theta = {theta:.4f} (par jour)")
    print(f"Rho = {rho:.4f} (pour 1% de taux d'intérêt)")


def plot_payoff(option_type: str, S: float, K: float, option_price: float):
    """
    Trace le graphique du profit/perte net de l'option à l'échéance.

    Args:
        option_type (str): Type de l'option ('C' ou 'P').
        S (float): Prix spot actuel du sous-jacent.
        K (float): Prix d'exercice de l'option.
        option_price (float): Prix calculé de l'option (coût initial).
    """
    # Une plage plus large pour bien voir le payoff, centrée autour du strike et du prix spot
    min_S_range = max(0, min(S, K) * 0.7)
    max_S_range = max(S, K) * 1.3
    S_payoff_range = np.linspace(min_S_range, max_S_range, 200)

    profit_loss_values = []
    for s_at_maturity in S_payoff_range:
        if option_type == "C":
            payoff_brut = max(0, s_at_maturity - K)
        elif option_type == "P":
            payoff_brut = max(0, K - s_at_maturity)

        profit_loss = payoff_brut - option_price
        profit_loss_values.append(profit_loss)

    plt.figure(figsize=(10, 6))
    plt.plot(
        S_payoff_range,
        profit_loss_values,
        label=f"Profit/Perte de l'option {option_type}",
        color="purple",
    )
    plt.axvline(x=K, color="green", linestyle="--", label=f"Prix d'exercice (K = {K})")
    plt.axhline(y=0, color="gray", linestyle="-")  # Ligne zéro pour le profit/perte

    # Marquer le point d'équilibre (breakeven point)
    if option_type == "C":
        breakeven_point = K + option_price
    elif option_type == "P":
        breakeven_point = K - option_price
        if breakeven_point < 0:  # Pour un put, le breakeven ne peut pas être négatif
            breakeven_point = 0

    plt.axvline(
        x=breakeven_point,
        color="orange",
        linestyle=":",
        label=f"Point d'équilibre ({breakeven_point:.2f})",
    )

    plt.title(
        f"Profit/Perte net de l'option {option_type} à l'échéance (Prix de l'option = {option_price:.2f})"
    )
    plt.xlabel("Prix du sous-jacent à l'échéance ($)")
    plt.ylabel("Profit/Perte ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
