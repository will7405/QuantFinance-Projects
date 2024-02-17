import numpy as np 
from scipy.stats import norm

# Functions for compyting d1 and d2 in BSM equation
d1 = lambda S, K, T, sigma, r : ( np.log(S/K) + (r + (sigma**2 / 2) )*T ) / ( sigma * np.sqrt(T))
d2 = lambda S, K, T, sigma, r : d1(S, K, T, sigma, r) - ( sigma * np.sqrt(T) )

# Price for call and put given by BSM equations
call_price = lambda S, K, T, sigma, r : ( S * norm.cdf(d1(S, K, T, sigma, r)) ) - ( K * np.exp(-1*r*T) * norm.cdf(d2(S, K, T, sigma, r)) )
put_price = lambda S, K, T, sigma, r : ( K * np.exp(-r*T) * norm.cdf(-d2(S, K, T, sigma, r)) ) - ( S * norm.cdf(-d1(S, K, T, sigma, r)) )


# Functions to calculate the greeks given by the BSM equations
delta_call = lambda S, K, T, sigma, r : norm.cdf(d1(S, K, T, sigma, r))
delta_put = lambda S, K, T, sigma, r : delta_call(S, K, T, sigma, r) - 1

gamma = lambda S, K, T, sigma, r : norm.pdf(d1(S, K, T, sigma, r)) / ( S * sigma * np.sqrt(T) )
vega = lambda S, K, T, sigma, r : S * norm.pdf(d1(S, K, T, sigma, r)) * np.sqrt(T) 

theta_call = lambda S, K, T, sigma, r : -1 * ( ( (-S * norm.pdf(d1(S, K, T, sigma, r)) * sigma) / (2 * np.sqrt(T)) ) - ( r * K * np.exp(-r*T) * norm.cdf(d2(S, K, T, sigma, r)) ) )
theta_put = lambda S, K, T, sigma, r : -1 * ( ( (-S * norm.pdf(d1(S, K, T, sigma, r)) * sigma) / (2 * np.sqrt(T)) ) + ( r * K * np.exp(-r*T) * norm.cdf(-d2(S, K, T, sigma, r)) ) )

rho_call = lambda S, K, T, sigma, r : K * T * np.exp(-r*T) * norm.cdf(d2(S, K, T, sigma, r))
rho_put = lambda S, K, T, sigma, r : -K * T * np.exp(-r*T) * norm.cdf(-d2(S, K, T, sigma, r))