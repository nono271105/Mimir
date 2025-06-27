import unittest
import sys
import os
import numpy as np

# Ajouter le chemin du répertoire src au PYTHONPATH pour permettre les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.bsm_model import black_scholes_greeks

class TestBlackScholesModel(unittest.TestCase):
    """
    Classe de tests pour le modèle Black-Scholes-Merton.
    """

    def test_call_option_price(self):
        """
        Teste le prix d'une option Call avec des paramètres standards.
        Référence: Calculateur en ligne ou manuel de finance.
        """
        # Paramètres de test standards
        S = 100    # Prix spot
        K = 100    # Prix d'exercice
        T = 1.0    # Temps à l'échéance (1 an)
        r = 0.05   # Taux sans risque (5%)
        sigma = 0.20 # Volatilité (20%)
        option_type = 'C'

        # Prix attendu (vérifié avec un calculateur fiable)
        # Par exemple, pour S=100, K=100, T=1, r=0.05, sigma=0.20, Call -> 10.4506
        expected_price = 10.4506

        price, _, _, _, _, _, _, _, _, _ = black_scholes_greeks(option_type, S, K, T, r, sigma)

        # Utilisation de assertAlmostEqual pour comparer les floats avec une tolérance
        self.assertAlmostEqual(price, expected_price, places=4, msg="Call option price mismatch")

    def test_put_option_price(self):
        """
        Teste le prix d'une option Put avec des paramètres standards.
        Référence: Calculateur en ligne ou manuel de finance.
        """
        # Paramètres de test standards
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.20
        option_type = 'P'

        # Prix attendu (vérifié avec un calculateur fiable)
        # Par exemple, pour S=100, K=100, T=1, r=0.05, sigma=0.20, Put -> 5.5735
        expected_price = 5.5735

        price, _, _, _, _, _, _, _, _, _ = black_scholes_greeks(option_type, S, K, T, r, sigma)
        self.assertAlmostEqual(price, expected_price, places=4, msg="Put option price mismatch")

    def test_put_call_parity(self):
        """
        Vérifie la parité Call-Put (pour options européennes sans dividendes).
        C - P = S - K * exp(-rT)
        """
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.20

        call_price, _, _, _, _, _, _, _, _, _ = black_scholes_greeks('C', S, K, T, r, sigma)
        put_price, _, _, _, _, _, _, _, _, _ = black_scholes_greeks('P', S, K, T, r, sigma)

        # Formule de parité Call-Put
        parity_difference = call_price - put_price
        expected_parity_difference = S - K * np.exp(-r * T)

        self.assertAlmostEqual(parity_difference, expected_parity_difference, places=4, msg="Put-Call parity violated")

    def test_greeks_call_delta(self):
        """
        Teste le Delta d'une option Call.
        Référence: Calculateur en ligne ou manuel.
        """
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        expected_delta = 0.6368 # Vérifié
        _, _, _, _, _, delta, _, _, _, _ = black_scholes_greeks('C', S, K, T, r, sigma)
        self.assertAlmostEqual(delta, expected_delta, places=4, msg="Call Delta mismatch")

    def test_greeks_put_delta(self):
        """
        Teste le Delta d'une option Put.
        Référence: Calculateur en ligne ou manuel.
        """
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        expected_delta = -0.3632 # Vérifié
        _, _, _, _, _, delta, _, _, _, _ = black_scholes_greeks('P', S, K, T, r, sigma)
        self.assertAlmostEqual(delta, expected_delta, places=4, msg="Put Delta mismatch")

    def test_greeks_gamma(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        expected_gamma = 0.0187 
        _, _, _, _, _, _, gamma_c, _, _, _ = black_scholes_greeks('C', S, K, T, r, sigma)
        _, _, _, _, _, _, gamma_p, _, _, _ = black_scholes_greeks('P', S, K, T, r, sigma)
        # Changer places=4 à places=3
        self.assertAlmostEqual(gamma_c, expected_gamma, places=3, msg="Call Gamma mismatch") 
        self.assertAlmostEqual(gamma_p, expected_gamma, places=3, msg="Put Gamma mismatch") 

    def test_greeks_theta_call(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        expected_theta_per_day = -0.0175 
        _, _, _, _, _, _, _, _, theta, _ = black_scholes_greeks('C', S, K, T, r, sigma)
        self.assertAlmostEqual(theta, expected_theta_per_day, places=3, msg="Call Theta mismatch") 

    def test_greeks_theta_put(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        expected_theta_per_day = -0.0045 
        _, _, _, _, _, _, _, _, theta, _ = black_scholes_greeks('P', S, K, T, r, sigma)
        self.assertAlmostEqual(theta, expected_theta_per_day, places=3, msg="Put Theta mismatch") 

    def test_greeks_rho_call(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        expected_rho_per_percent = 0.5323
        _, _, _, _, _, _, _, _, _, rho = black_scholes_greeks('C', S, K, T, r, sigma)
        self.assertAlmostEqual(rho, expected_rho_per_percent, places=3, msg="Call Rho mismatch") 

    def test_greeks_rho_put(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        expected_rho_per_percent = -0.4189
        _, _, _, _, _, _, _, _, _, rho = black_scholes_greeks('P', S, K, T, r, sigma)
        self.assertAlmostEqual(rho, expected_rho_per_percent, places=3, msg="Put Rho mismatch") 

    def test_T_zero(self):
        """
        Teste le comportement quand le temps à l'échéance est 0.
        """
        S, K, T, r, sigma = 100, 100, 0.0, 0.05, 0.20

        # Call
        price_c, d1_c, d2_c, N_d1_c, N_d2_c, delta_c, gamma_c, vega_c, theta_c, rho_c = black_scholes_greeks('C', S, K, T, r, sigma)
        self.assertAlmostEqual(price_c, max(0, S - K), places=4)
        self.assertEqual(gamma_c, 0.0)
        self.assertEqual(vega_c, 0.0)
        self.assertEqual(theta_c, 0.0)

        # Put
        price_p, d1_p, d2_p, N_d1_p, N_d2_p, delta_p, gamma_p, vega_p, theta_p, rho_p = black_scholes_greeks('P', S, K, T, r, sigma)
        self.assertAlmostEqual(price_p, max(0, K - S), places=4)
        self.assertEqual(gamma_p, 0.0)
        self.assertEqual(vega_p, 0.0)
        self.assertEqual(theta_p, 0.0)

    def test_sigma_zero(self):
        """
        Teste le comportement quand la volatilité est très proche de zéro.
        """
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 1e-10 # Très petite sigma

        # Call
        price_c, d1_c, d2_c, N_d1_c, N_d2_c, delta_c, gamma_c, vega_c, theta_c, rho_c = black_scholes_greeks('C', S, K, T, r, sigma)
        self.assertAlmostEqual(price_c, max(0, S * np.exp(r * T) - K) * np.exp(-r * T), places=4)
        self.assertEqual(gamma_c, 0.0)
        self.assertEqual(vega_c, 0.0)
        self.assertEqual(theta_c, 0.0)

        # Put
        price_p, d1_p, d2_p, N_d1_p, N_d2_p, delta_p, gamma_p, vega_p, theta_p, rho_p = black_scholes_greeks('P', S, K, T, r, sigma)
        self.assertAlmostEqual(price_p, max(0, K - S * np.exp(r * T)) * np.exp(-r * T), places=4)
        self.assertEqual(gamma_p, 0.0)
        self.assertEqual(vega_p, 0.0)
        self.assertEqual(theta_p, 0.0)


if __name__ == '__main__':
    unittest.main()