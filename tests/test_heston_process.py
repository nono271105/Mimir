# tests/test_heston_process.py
import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.heston.process import generate_heston_paths


class TestHestonProcess(unittest.TestCase):

    def setUp(self):
        # Paramètres de test typiques pour le modèle de Heston
        self.S0 = 100.0  # Prix initial de l'actif
        self.V0 = 0.04  # Variance initiale (vol = 20%)
        self.kappa = 2.0  # Vitesse de retour à la moyenne
        self.theta = 0.04  # Variance moyenne à long terme
        self.xi = 0.1  # Volatilité de la volatilité
        self.rho = -0.5  # Corrélation
        self.T = 1.0  # Temps à l'échéance (1 an)
        self.r = 0.05  # Taux sans risque
        self.N_steps = 252  # Nombre de pas de temps (jours de trading sur 1 an)
        self.N_simulations = (
            10000  # Nombre de simulations pour des résultats statistiques fiables
        )

    def test_output_dimensions(self):
        """Vérifie que les tableaux de sortie ont les dimensions attendues."""
        paths_S, paths_V = generate_heston_paths(
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
        self.assertEqual(paths_S.shape, (self.N_simulations, self.N_steps + 1))
        self.assertEqual(paths_V.shape, (self.N_simulations, self.N_steps + 1))
        print(f"Test dimensions: OK. S_shape={paths_S.shape}, V_shape={paths_V.shape}")

    def test_initial_conditions(self):
        """Vérifie que les conditions initiales S0 et V0 sont correctement fixées."""
        paths_S, paths_V = generate_heston_paths(
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
        np.testing.assert_array_equal(paths_S[:, 0], self.S0)
        np.testing.assert_array_equal(paths_V[:, 0], self.V0)
        print(f"Test conditions initiales: OK. S0={paths_S[0,0]}, V0={paths_V[0,0]}")

    def test_variance_positivity(self):
        """Vérifie que la variance reste toujours non-négative."""
        paths_S, paths_V = generate_heston_paths(
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
        self.assertTrue(
            np.all(paths_V >= 0.0), "La variance doit toujours être non-négative."
        )
        print(f"Test positivité variance: OK. Min V = {np.min(paths_V):.4f}")

    def test_no_nan_inf(self):
        """Vérifie qu'il n'y a pas de valeurs NaN ou infinies dans les chemins."""
        paths_S, paths_V = generate_heston_paths(
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
        self.assertFalse(np.any(np.isnan(paths_S)), "paths_S contient des NaN.")
        self.assertFalse(np.any(np.isinf(paths_S)), "paths_S contient des Inf.")
        self.assertFalse(np.any(np.isnan(paths_V)), "paths_V contient des NaN.")
        self.assertFalse(np.any(np.isinf(paths_V)), "paths_V contient des Inf.")
        print("Test NaN/Inf: OK.")


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
