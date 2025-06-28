# tests/test_exotic_pricer.py
import unittest
import numpy as np
import sys
import os

# Ensure the project root is in sys.path for module discovery
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.insert(0, project_root)

# Corrected imports for your project structure (assuming 'src' is directly under project_root)
from src.models.heston.process import generate_heston_paths
from core.monte_carlo_pricer import (
    run_monte_carlo,
)  # Assurez-vous que core est aussi sous src si c'était votre structure
from src.models.exotic.payoffs import (
    calculate_barrier_payoff,
    calculate_asian_payoff,
    calculate_digital_payoff,
)
from src.models.exotic.pricing import (
    price_heston_barrier_option,
    price_heston_asian_option,
    price_heston_digital_option,
)


class TestExoticOptionPricing(unittest.TestCase):

    def setUp(self):
        # Paramètres du modèle Heston configurés pour approcher Black-Scholes
        self.S0 = 50.0
        self.V0 = 0.4**2  # Sigma^2 de Black-Scholes (0.16)
        self.kappa = 2.0  # Vitesse de retour à la moyenne (peut être élevée quand xi=0)
        self.theta = 0.4**2  # Variance longue terme = Sigma^2 (0.16)
        self.xi = 0.0  # Volatilité de la volatilité = 0 (CLÉ pour constante)
        self.rho = 0.0  # Corrélation (irrelevant quand xi=0)
        self.T = 1.0
        self.r = 0.1
        self.N_steps = 250  # 250 observations pour 1 an, comme dans l'exemple du Hull
        self.N_simulations = (
            500000  # Augmenter pour une meilleure précision Monte Carlo
        )

    # ... (autres tests, inchangés) ...

    def test_price_heston_asian_option_benchmark(self):
        """
        Teste le pricing complet d'une option asiatique Heston configurée en Black-Scholes
        contre le benchmark du Hull (250 observations).
        """
        price = price_heston_asian_option(
            S0=self.S0,
            V0=self.V0,
            kappa=self.kappa,
            theta=self.theta,
            xi=self.xi,
            rho=self.rho,
            T=self.T,
            r=self.r,
            N_steps=self.N_steps,
            N_simulations=self.N_simulations,
            K=50.0,
            option_type="C",
        )

        expected_price = 5.563410942477913
        delta = 0.05  # Tolérance de 5 centimes par exemple

        print(f"Prix Asiatique Monte Carlo (Heston-BS): {price:.4f}")
        print(f"Prix Asiatique Hull (Benchmark): {expected_price:.2f}")

        self.assertAlmostEqual(
            price,
            expected_price,
            delta=delta,
            msg=f"Le prix Monte Carlo {price:.4f} ne correspond pas au benchmark {expected_price:.2f}",
        )
