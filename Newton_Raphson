import numpy as np
import scipy.stats as si
import math

# Helper functions for Black-Scholes 
def d1(S, K, r, sigma, T):
    return (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))

def d2(S, K, r, sigma, T):
    return d1(S, K, r, sigma, T) - sigma*np.sqrt(T)

#* computation of the market call/put price 
def call_price(S, K, r, sigma, T):
    return S*si.norm.cdf(d1(S, K, r, sigma, T)) - K*np.exp(-r*T)*si.norm.cdf(d2(S, K, r, sigma, T))

def put_price(S, K, r, sigma, T):
    return K*np.exp(-r*T)*si.norm.cdf(-d2(S, K, r, sigma, T)) - S*si.norm.cdf(-d1(S, K, r, sigma, T))

#* computation of the vega 
def call_vega(S, K, r, sigma, T):
    return S*np.sqrt(T)*si.norm.pdf(d1(S, K, r, sigma, T))

def put_vega(S, K, r, sigma, T):
    return S*np.sqrt(T)*si.norm.pdf(d1(S, K, r, sigma, T))

# Implied volatility using Newton-Raphson for Call
def call_implied_volatility(S, K, T, C_market, r, sigma_init=0.2, tolerance=1e-6, max_iter=100):
    sigma = sigma_init
    for i in range(max_iter):
        C_bs = call_price(S, K, r, sigma, T)
        vega = call_vega(S, K, r, sigma, T)
        error = C_bs - C_market
        
        if abs(error) < tolerance:
            print(f"Converged in {i+1} iterations. Error: {error:.8f}")
            return sigma
        
        sigma = sigma - error / vega
    
    print(f"Max iterations reached. Final error: {error:.8f}")
    return sigma

# Implied volatility using Newton-Raphson for Put
def put_implied_volatility(S, K, T, P_market, r, sigma_init=0.2, tolerance=1e-6, max_iter=100):
    sigma = sigma_init
    for i in range(max_iter):
        P_bs = put_price(S, K, r, sigma, T)
        vega = put_vega(S, K, r, sigma, T)
        error = P_bs - P_market
        
        if abs(error) < tolerance:
            print(f"Converged in {i+1} iterations. Error: {error:.8f}")
            return sigma
        
        sigma = sigma - error / vega
    
    print(f"Max iterations reached. Final error: {error:.8f}")
    return sigma

# Example usage
S = 100           # Spot price
K = 100           # Strike price
r = 0.05          # Risk-free rate
T = 1.0           # Time to expiration (1 year)
C_market = 10.45  # Market call price
P_market = 5.57   # Market put price

iv_call = call_implied_volatility(S, K, T, C_market, r, sigma_init=0.2)
iv_put = put_implied_volatility(S, K, T, P_market, r, sigma_init=0.2)

print(f"Call Implied Volatility: {iv_call:.6f} ({iv_call*100:.2f}%)")
print(f"Put Implied Volatility: {iv_put:.6f} ({iv_put*100:.2f}%)")
