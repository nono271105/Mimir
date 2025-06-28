# tests/test_monte_carlo_pricer.py
import unittest
import sys
import os
import numpy as np
from numba import njit

# Ajouter les chemins des répertoires parents pour que Python puisse trouver les modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importation des fonctions nécessaires
from src.models.heston.process import generate_heston_paths
from core.monte_carlo_pricer import run_monte_carlo
from src.models.heston.heston_model import (
    heston_price,
)  # Pour obtenir un prix de référence analytique


# --- Fonctions de Payoff pour les tests (compatibles njit) ---
@njit
def european_call_payoff(path_prices: np.ndarray, K: float) -> float:
    """Calcule le payoff d'une option Call européenne à l'échéance."""
    final_price = path_prices[-1]
    return np.maximum(0.0, final_price - K)


@njit
def european_put_payoff(path_prices: np.ndarray, K: float) -> float:
    """Calcule le payoff d'une option Put européenne à l'échéance."""
    final_price = path_prices[-1]
    return np.maximum(0.0, K - final_price)


class TestMonteCarloPricer(unittest.TestCase):

    def setUp(self):
        # Paramètres pour la simulation Heston (utilisés pour générer les chemins)
        self.S0 = 100.0  # Prix initial de l'actif
        self.V0 = 0.04  # Variance initiale (équivalent à une volatilité de 20%)
        self.kappa = 2.0  # Vitesse de retour à la moyenne de la variance
        self.theta = 0.04  # Variance moyenne à long terme
        self.xi = 0.1  # Volatilité de la volatilité (sera passée comme 'sigma' à heston_price)
        self.rho = -0.5  # Corrélation
        self.T = 1.0  # Temps à l'échéance (1 an)
        self.r = 0.05  # Taux sans risque
        self.N_steps = 252  # Nombre de pas de temps
        self.N_simulations = (
            100000  # Nombre de simulations élevé pour une meilleure convergence MC
        )

        # Paramètres d'option pour les tests (Strike)
        self.K = 100.0

    def test_european_call_pricing(self):
        """
        Teste le pricing d'un Call Européen via Monte Carlo en comparant
        avec le prix du modèle de Heston analytique.
        """
        print("\n--- Test Pricing Call Européen avec Monte Carlo ---")

        # 1. Générer les chemins Heston pour le sous-jacent
        paths_S, _ = generate_heston_paths(
            self.S0,
            self.V0,
            self.kappa,
            self.theta,
            self.xi,
            self.rho,
            self.T,
            self.r,
            self.N_steps,
            self.N_simulations,
        )

        # 2. Pricer l'option Call Européenne avec le moteur Monte Carlo
        mc_price_call = run_monte_carlo(
            paths_S, self.T, self.r, european_call_payoff, self.K
        )
        print(f"Prix Call Monte Carlo: {mc_price_call:.4f}")

        # 3. Obtenir le prix de référence avec la formule analytique de Heston
        # L'argument 'sigma' de votre heston_price reçoit ici la valeur de 'xi' (vol-of-vol)
        ref_price_call = heston_price(
            S=self.S0,
            K=self.K,
            T=self.T,
            r=self.r,
            kappa=self.kappa,
            theta=self.theta,
            sigma=self.xi,  # 'sigma' prend la valeur de 'xi'
            rho=self.rho,
            v0=self.V0,
            option_type="C",
        )
        print(f"Prix Call Heston Analytique: {ref_price_call:.4f}")

        self.assertAlmostEqual(
            mc_price_call,
            ref_price_call,
            delta=0.5,
            msg="Le prix MC du Call ne converge pas suffisamment vers le prix analytique Heston.",
        )
        print("Test Pricing Call Européen: OK.")

    def test_european_put_pricing(self):
        """
        Teste le pricing d'un Put Européen via Monte Carlo en comparant
        avec le prix du modèle de Heston analytique.
        """
        print("\n--- Test Pricing Put Européen avec Monte Carlo ---")

        # 1. Générer les chemins Heston pour le sous-jacent
        paths_S, _ = generate_heston_paths(
            self.S0,
            self.V0,
            self.kappa,
            self.theta,
            self.xi,
            self.rho,
            self.T,
            self.r,
            self.N_steps,
            self.N_simulations,
        )

        # 2. Pricer l'option Put Européenne avec le moteur Monte Carlo
        mc_price_put = run_monte_carlo(
            paths_S, self.T, self.r, european_put_payoff, self.K
        )
        print(f"Prix Put Monte Carlo: {mc_price_put:.4f}")

        # 3. Obtenir le prix de référence avec la formule analytique de Heston
        ref_price_put = heston_price(
            S=self.S0,
            K=self.K,
            T=self.T,
            r=self.r,
            kappa=self.kappa,
            theta=self.theta,
            sigma=self.xi,  # 'sigma' prend la valeur de 'xi'
            rho=self.rho,
            v0=self.V0,
            option_type="P",
        )
        print(f"Prix Put Heston Analytique: {ref_price_put:.4f}")

        self.assertAlmostEqual(
            mc_price_put,
            ref_price_put,
            delta=0.5,
            msg="Le prix MC du Put ne converge pas suffisamment vers le prix analytique Heston.",
        )
        print("Test Pricing Put Européen: OK.")


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
