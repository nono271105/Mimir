# tests/test_heston_model.py
import unittest
import sys
import os
import numpy as np

# Ajouter le chemin du répertoire parent pour pouvoir importer les modules des modèles
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importer la fonction heston_price que nous venons d'implémenter
from src.models.heston.heston_model import heston_price


class TestHestonModel(unittest.TestCase):

    def test_heston_call_example_1(self):
        """
        Test d'une option Call européenne Heston avec des paramètres de référence.
        Les valeurs de référence pour Heston sont sensibles et doivent être vérifiées
        avec des calculateurs ou articles académiques fiables.
        """
        S = 100.0  # Prix spot
        K = 100.0  # Prix d'exercice
        T = 1.0  # Temps à l'échéance (1 an)
        r = 0.03  # Taux sans risque
        kappa = 1.5  # Vitesse de retour à la moyenne
        theta = (
            0.04  # Variance moyenne à long terme (volatility^2 = 0.04 -> vol = 0.20)
        )
        sigma = 0.3  # Vol-of-vol
        rho = -0.5  # Corrélation
        v0 = 0.04  # Variance initiale (volatility^2 = 0.04 -> vol = 0.20)
        option_type = "C"
        expected_price = 9.2528  # Valeur arrondie d'un exemple validé.

        calculated_price = heston_price(
            S, K, T, r, kappa, theta, sigma, rho, v0, option_type
        )

        # Nous utilisons 'places=2' ou 'places=3' car l'intégration numérique
        # peut entraîner de légères variations de précision.
        self.assertAlmostEqual(calculated_price, expected_price, places=2)
        print(
            f"Test Heston Call 1: Calculé={calculated_price:.3f}, Attendu={expected_price:.3f}"
        )

    def test_heston_put_example_1(self):
        """
        Test d'une option Put européenne Heston avec les mêmes paramètres que le Call 1.
        Utilisation de la parité Call-Put pour valider le prix du Put.
        """
        S = 100.0
        K = 100.0
        T = 1.0
        r = 0.03
        kappa = 1.5
        theta = 0.04
        sigma = 0.3
        rho = -0.5
        v0 = 0.04
        option_type = "P"
        expected_price = 6.2973  # Arrondi.

        calculated_price = heston_price(
            S, K, T, r, kappa, theta, sigma, rho, v0, option_type
        )

        self.assertAlmostEqual(calculated_price, expected_price, places=2)
        print(
            f"Test Heston Put 1: Calculé={calculated_price:.3f}, Attendu={expected_price:.3f}"
        )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
