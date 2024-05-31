import numpy as np
from scipy.stats import norm

S = 100
r = 0.1
vol = 0.2

# If T0 = 0, then m and M are automatically equal to the current stock price S
def lookback_call(S, T, r, vol):
    # To be consistent with the paper
    m = S
    d2 = (np.log(S/m) + r*T + (1/2)*vol**2*T) / (vol*np.sqrt(T))
    C = S*norm.cdf(d2) - np.exp(-r*T)*m*norm.cdf(d2 - vol*np.sqrt(T)) + np.exp(-r*T)*(vol**2 / (2*r))*S*((S/m)**(-2*r/vol**2) * norm.cdf(-d2 + (2*r/vol)*np.sqrt(T)) - np.exp(r*T)*norm.cdf(-d2))
    return C

def lookback_put(S, T, r, vol):
    # To be consistent with the paper
    M = S
    d1 = (np.log(S/M) + r*T + (1/2)*vol**2*T) / (vol*np.sqrt(T))
    C = -S*norm.cdf(-d1) + np.exp(-r*T)*M*norm.cdf(-d1 + vol*np.sqrt(T)) + np.exp(-r*T)*(vol**2 / (2*r))*S*(-(S/M)**(-2*r/vol**2) * norm.cdf(d1 - (2*r/vol)*np.sqrt(T)) + np.exp(r*T)*norm.cdf(d1))
    return C

print("Expected: 8.95, Actual Output: {}".format(lookback_call(S, 0.25, r, vol)))
print("Expected: 6.98, Actual Output: {}".format(lookback_put(S, 0.25, r, vol)))

S = 100
r = 0.1
vol = 0.3
T = 0.5
m = 5
# Actual put price should be 10.06425  Continuous put is 15.35

beta1 = 0.5826
# First and second order approximations
def lookback_discrete_call_first(S, T, r, vol, m):
    return (lookback_call(S, T, r, vol) + S) * np.exp(-beta1*vol*np.sqrt(T/m)) - S
    
def lookback_discrete_call_second(S, T, r, vol, m):
    return -(np.exp(beta1*vol*np.sqrt(T/m))*lookback_call(S*np.exp(-beta1*vol*np.sqrt(T/m)), T, r, vol) + (np.exp(beta1*vol*np.sqrt(T/m))-1)*S)

def lookback_discrete_put_first(S, T, r, vol, m):
    return (lookback_put(S, T, r, vol) + S) * np.exp(-beta1*vol*np.sqrt(T/m)) - S

def lookback_discrete_put_second(S, T, r, vol, m):
    return np.exp(-beta1*vol*np.sqrt(T/m))*lookback_put(S*np.exp(beta1*vol*np.sqrt(T/m)), T, r, vol) + S*np.exp(-beta1*vol*np.sqrt(T/m)) - S

print("continous put price: ", lookback_put(S,T,r,vol))
print("first order approx => Expected: 9.15, Actual Output: ", lookback_discrete_put_first(S,T,r,vol,m))
print("second order approx => Expected: 10.18, Actual Output: ", lookback_discrete_put_second(S,T,r,vol,m))

def mc_lookback_put(S, T, r, vol, m):
    # number of simulations (nice if multiple of m)
    n = 100000

    # Number of total steps. If k is a multiple of m, it makes code much cleaner
    k = m
    
    # time step
    dt = T/k

    # GBM evolution of a stock
    simulation = np.ones((n,k+1))
    simulation[:,0] = S
    for i in range(1,k+1):
        simulation[:,i] = simulation[:,i-1]*np.exp((r-(vol**2)/2)*dt + vol*np.sqrt(dt)*np.random.normal(0,1,n))

    # Get the possible exercies prices
    lookbackDates = simulation[:, int(k/m)-1::int(k/m)]

    # Get the maximum possible exercies price in each simulation
    lookbackMax = np.amax(lookbackDates, axis=1)
    
    payoffs = np.maximum(lookbackMax - simulation[:,-1],0)

    return np.exp(-r*T)*np.average(payoffs)

print(mc_lookback_put(S,T,r,vol,m))