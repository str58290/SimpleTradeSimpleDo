from scipy.optimize import brentq
import scipy.stats as si
import numpy as np

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

def implied_volatility_brentq(option_price, S, K, T, r, option_type='call'):
    """Calculate IV using Brent's method (bracketing algorithm)"""
    
    def objective(sigma):
        if option_type == 'call':
            return call_price(S, K, r, sigma, T) - option_price
        else:
            return put_price(S, K, r, sigma, T) - option_price
    
    # Search bounds: volatility typically between 0.1% and 500%
    iv = brentq(objective, a=1e-6, b=5.0, xtol=1e-8, rtol=1e-8)
    return iv

# Example usage
iv_call_brentq = implied_volatility_brentq(10.45, 100, 100, 1.0, 0.05, 'call')
print(f"IV (Brent's method): {iv_call_brentq:.6f}")
