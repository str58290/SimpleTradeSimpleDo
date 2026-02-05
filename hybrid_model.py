import numpy as np
import scipy.stats as si
import math
# Helper functions for Black-Scholes
def d1(S, K, r, sigma, T):
    return (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))

def d2(S, K, r, sigma, T):
    return d1(S, K, r, sigma, T) - sigma*np.sqrt(T)

def call_price(S, K, r, sigma, T):
    return S*si.norm.cdf(d1(S, K, r, sigma, T)) - K*np.exp(-r*T)*si.norm.cdf(d2(S, K, r, sigma, T))

def put_price(S, K, r, sigma, T):
    return K*np.exp(-r*T)*si.norm.cdf(-d2(S, K, r, sigma, T)) - S*si.norm.cdf(-d1(S, K, r, sigma, T))

def call_vega(S, K, r, sigma, T):
    return S*np.sqrt(T)*si.norm.pdf(d1(S, K, r, sigma, T))

def put_vega(S, K, r, sigma, T):
    return S*np.sqrt(T)*si.norm.pdf(d1(S, K, r, sigma, T))

def implied_volatility_hybrid(option_price, S, K, T, r, option_type='call', epsilon=1e-8):
    """
    Hybrid Newton-Raphson + Bisection for IV calculation.
    Uses NR for fast convergence, falls back to bisection if needed.
    """
    
    def bs_price(sigma, option_type):
        if option_type == 'call':
            return call_price(S, K, r, sigma, T)
        else:
            return put_price(S, K, r, sigma, T)
    
    def vega_func(sigma):
        d1_val = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return S * np.sqrt(T) * si.norm.pdf(d1_val)
    
    # Start with Newton-Raphson
    sigma = 0.2
    for nr_iter in range(50):
        price_est = bs_price(sigma, option_type)
        error = price_est - option_price
        
        if abs(error) < epsilon:
            return sigma
        
        vega = vega_func(sigma)
        if abs(vega) < 1e-10:  # Vega too small, switch to bisection
            break
        
        sigma_new = sigma - error / vega
        
        # Safety: if sigma goes negative or extremely high, switch to bisection
        if sigma_new < 0 or sigma_new > 10:
            break
        
        sigma = sigma_new
    
    # Bisection fallback
    low, high = 0.0001, 5.0
    for bs_iter in range(100):
        mid = (low + high) / 2
        price_mid = bs_price(mid, option_type)
        error_mid = price_mid - option_price
        
        if abs(error_mid) < epsilon:
            return mid
        
        if error_mid < 0:
            low = mid
        else:
            high = mid
    
    return (low + high) / 2

S = 100      # spot
K = 100      # strike
T = 1.0      # time to expiry (years)
r = 0.05     # risk‑free rate
C_mkt = 10.45  # market call price

iv = implied_volatility_hybrid(C_mkt, S, K, T, r, option_type='call')
print("Implied volatility:", iv)
print("Implied volatility (%):", iv * 100)
