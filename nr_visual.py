import matplotlib.pyplot as plt
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

def calculate_implied_volatility(option_price, S, K, T, r, option_type='call'):
    epsilon = 1e-6
    sigma = 0.5  # Initial guess
    max_iterations = 100
    iterations = 0
    
    iteration_values = []
    iv_values = []
    
    while True:
        d1_val = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        vega = S * si.norm.pdf(d1_val) * math.sqrt(T)
        
        if option_type == 'call':
            option_price_estimate = call_price(S, K, r, sigma, T)
        else:
            option_price_estimate = put_price(S, K, r, sigma, T)
        
        iteration_values.append(iterations)
        iv_values.append(sigma)
        
        error = option_price - option_price_estimate
        
        if abs(error) < epsilon or iterations >= max_iterations:
            break
        
        sigma += error / vega
        iterations += 1
    
    return sigma, iteration_values, iv_values

# Example usage with plotting
iv, iters, iv_vals = calculate_implied_volatility(10.45, 100, 100, 1.0, 0.05, 'call')

plt.figure(figsize=(10, 6))
plt.plot(iters, iv_vals, marker='o', linestyle='-', linewidth=2)
plt.xlabel('Iterations')
plt.ylabel('Implied Volatility')
plt.title('Newton-Raphson Convergence: Implied Volatility')
plt.grid(True)
plt.show()

print(f"Final Implied Volatility: {iv:.6f}")
