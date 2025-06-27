import unittest
import sys
import os
import numpy as np 


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.binomial_model import binomial_option_pricing

class TestBinomialModel(unittest.TestCase):
    """
    Classe de tests pour le modèle binomial de Cox-Ross-Rubinstein (CRR).
    """

    def test_call_option_price(self):
        """
        Teste le prix d'une option Call avec le modèle binomial.
        Référence: Calculateur en ligne
        """
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.20
        N = 500 # Nombre de pas, plus N est grand, plus le prix binomial s'approche de BSM
        expected_price = 10.4465 # Cette valeur est très proche du BSM, ce qui est attendu.

        price = binomial_option_pricing('C', S, K, T, r, sigma, N)
        self.assertAlmostEqual(price, expected_price, places=3, msg="Binomial Call option price mismatch")

    def test_put_option_price(self):
        """
        Teste le prix d'une option Put avec le modèle binomial.
        Référence: Calculateur en ligne ou manuel de finance.
        """
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.20
        N = 100
        expected_price = 5.5535 

        price = binomial_option_pricing('P', S, K, T, r, sigma, N)
        self.assertAlmostEqual(price, expected_price, places=3, msg="Binomial Put option price mismatch")

    def test_T_zero(self):
        """
        Teste le comportement quand le temps à l'échéance est 0.
        Le prix devrait être le payoff intrinsèque.
        """
        S, K, T, r, sigma, N = 100, 100, 0.0, 0.05, 0.20, 10

        # Call
        price_c = binomial_option_pricing('C', S, K, T, r, sigma, N)
        self.assertAlmostEqual(price_c, max(0, S - K), places=4)

        # Put
        price_p = binomial_option_pricing('P', S, K, T, r, sigma, N)
        self.assertAlmostEqual(price_p, max(0, K - S), places=4)

    def test_N_one(self):
        """
        Teste le modèle avec un seul pas (N=1) pour une vérification simplifiée.
        """
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.20
        N = 1 # Un seul pas

        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        q = (np.exp(r * dt) - d) / (u - d)

        # Prix du sous-jacent à t=T (2 nœuds)
        Su = S * u
        Sd = S * d

        # Payoffs Call à t=T
        call_payoff_u = max(0, Su - K)
        call_payoff_d = max(0, Sd - K)
        # Prix Call à t=0
        expected_call_price = np.exp(-r * dt) * (q * call_payoff_u + (1 - q) * call_payoff_d)

        # Payoffs Put à t=T
        put_payoff_u = max(0, K - Su)
        put_payoff_d = max(0, K - Sd)
        # Prix Put à t=0
        expected_put_price = np.exp(-r * dt) * (q * put_payoff_u + (1 - q) * put_payoff_d)

        calculated_call_price = binomial_option_pricing('C', S, K, T, r, sigma, N)
        calculated_put_price = binomial_option_pricing('P', S, K, T, r, sigma, N)

        self.assertAlmostEqual(calculated_call_price, expected_call_price, places=4, msg="Binomial Call N=1 mismatch")
        self.assertAlmostEqual(calculated_put_price, expected_put_price, places=4, msg="Binomial Put N=1 mismatch")

    def test_N_invalid(self):
        """
        Teste la gestion d'erreur pour N invalide.
        """
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20

        with self.assertRaises(ValueError):
            binomial_option_pricing('C', S, K, T, r, sigma, 0) # N=0
        with self.assertRaises(ValueError):
            binomial_option_pricing('P', S, K, T, r, sigma, -5) # N négatif

    def test_q_out_of_bounds(self):
        """
        Teste la gestion d'erreur quand la probabilité risque-neutre q est hors bornes.
        Cela peut arriver avec des paramètres extrêmes.
        """
        # r très grand, sigma très petit
        S, K, T, r, sigma, N = 100, 100, 0.1, 10.0, 0.001, 10

        with self.assertRaises(ValueError):
            binomial_option_pricing('C', S, K, T, r, sigma, N)

        # r très petit (négatif), sigma très petit
        S, K, T, r, sigma, N = 100, 100, 0.1, -0.5, 0.001, 10
        with self.assertRaises(ValueError):
            binomial_option_pricing('P', S, K, T, r, sigma, N)

if __name__ == '__main__':
    unittest.main()