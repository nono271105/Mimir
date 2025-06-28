import sys
import os
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from models.bjerksund_stensland_model import bjerksund_stensland_2002

# --- Test case: test_call_option_different_params ---
S, K, T, r, sigma, q = 50, 55, 0.5, 0.02, 0.3, 0.01
calculated_price_call_diff_params = bjerksund_stensland_2002('C', S, K, T, r, sigma, q)
print(f"test_call_option_different_params: {calculated_price_call_diff_params}")

# --- Test case: test_call_option_no_dividends ---
S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.20, 0.0
calculated_price_call_no_div = bjerksund_stensland_2002('C', S, K, T, r, sigma, q)
print(f"test_call_option_no_dividends: {calculated_price_call_no_div}")

# --- Test case: test_call_option_with_dividends ---
S, K, T, r, sigma, q = 90, 85, 0.5, 0.04, 0.25, 0.02
calculated_price_call_with_div = bjerksund_stensland_2002('C', S, K, T, r, sigma, q)
print(f"test_call_option_with_dividends: {calculated_price_call_with_div}")

# --- Test case: test_put_option_different_params ---
S, K, T, r, sigma, q = 50, 55, 0.5, 0.02, 0.3, 0.01
calculated_price_put_diff_params = bjerksund_stensland_2002('P', S, K, T, r, sigma, q)
print(f"test_put_option_different_params: {calculated_price_put_diff_params}")

# --- Test case: test_put_option_no_dividends ---
S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.20, 0.0
calculated_price_put_no_div = bjerksund_stensland_2002('P', S, K, T, r, sigma, q)
print(f"test_put_option_no_dividends: {calculated_price_put_no_div}")

# --- Test case: test_put_option_with_dividends ---
S, K, T, r, sigma, q = 90, 95, 0.5, 0.04, 0.25, 0.02
calculated_price_put_with_div = bjerksund_stensland_2002('P', S, K, T, r, sigma, q)
print(f"test_put_option_with_dividends: {calculated_price_put_with_div}")