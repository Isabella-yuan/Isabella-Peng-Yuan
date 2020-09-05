'''
This module defined functions of 1.vanilla (call/put) option, 2.Cash-or-nothing (call/put) option,
and 3.Asset-or-nothing (call/put) option under 1.Black-Scholes Model, 2.Bachelier Model, 3.Black76 Model, and 4.Displaced-Diffusion Model.
S_0: the spot price at the begining
K: strike price
r: risk-free rate
sigma: volatility of underlying asset, standard deivation of the asset return
T: time to maturity year based
F_0: forward price at the begining
'''
import numpy as np
import scipy.stats as ss


#Black-Scholes Model ps.d1,d2 in all functions come from call option
#1 vanilla option
def Black_Scholes_call(S_0, K, T, r, sigma):
    '''European call option pricing fomula under Black-Scholes Model'''
    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    V_0 = S_0 * ss.norm.cdf(d1) - K * np.exp(-r * T) * ss.norm.cdf(d2)
    return V_0


def Black_Scholes_put(S_0, K, T, r, sigma):
    '''European put option pricing fomula under Black-Scholes Model'''
    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    V_0 = -S_0 * ss.norm.cdf(-d1) + K * np.exp(-r * T) * ss.norm.cdf(-d2)
    return V_0


#2 digital cash-or-nothing option
def Black_Scholes_dcon_call(S_0, K, T, r, sigma):
    '''digital cash-or-nothing call option pricing fomula under Black-Scholes Model'''
    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    V_0 = np.exp(-r * T) * ss.norm.cdf(d2)
    return V_0


def Black_Scholes_dcon_put(S_0, K, T, r, sigma):
    '''digital cash-or-nothing put option pricing fomula under Black-Scholes Model'''
    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    V_0 = np.exp(-r * T) * ss.norm.cdf(-d2)
    return V_0


#3 digital asset or nothing
def Black_Scholes_daon_call(S_0, K, T, r, sigma):
    '''digital asset-or-nothing call option pricing fomula under Black-Scholes Model'''
    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    V_0 = S_0 * ss.norm.cdf(d1)
    return V_0


def Black_Scholes_daon_put(S_0, K, T, r, sigma):
    '''digital asset-or-nothing put option pricing fomula under Black-Scholes Model'''
    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    V_0 = S_0 * ss.norm.cdf(-d1)
    return V_0


#Bachelier Model ps.d1 in all functions come from call option
#1 vanilla option
def Bachelier_call(S_0, K, T, sigma):
    '''European call option pricing fomula under Bachelier Model'''
    d1 = (S_0 - K) / (sigma * np.sqrt(T) * S_0)
    V_0 = (S_0 -K) * ss.norm.cdf(d1) + S_0 * sigma * np.sqrt(T) * ss.norm.pdf(d1)
    return V_0


def Bachelier_put(S_0, K, T, sigma):
    '''European put option pricing fomula under Bachelier Model'''
    d1 = (S_0 - K) / (sigma * np.sqrt(T) * S_0)
    V_0 = -(S_0 -
            K) * ss.norm.cdf(-d1) + S_0 * sigma * np.sqrt(T) * ss.norm.pdf(-d1)
    return V_0


#2 digital cash or nothing
def Bachelier_dcon_call(S_0, K, T, sigma):
    '''digital cash-or-nothing call option pricing fomula under Bachelier Model'''
    d1 = (S_0 - K) / (sigma * np.sqrt(T) * S_0)
    V_0 = ss.norm.cdf(d1)
    return V_0


def Bachelier_dcon_put(S_0, K, T, sigma):
    '''digital cash-or-nothing put option pricing fomula under Bachelier Model'''
    d1 = (S_0 - K) / (sigma * np.sqrt(T) * S_0)
    V_0 = ss.norm.cdf(-d1)
    return V_0


#3 digital asset or nothing
def Bachelier_daon_call(S_0, K, T, sigma):
    '''digital asset-or-nothing call option pricing fomula under Bachelier Model'''
    d1 = (S_0 - K) / (sigma * np.sqrt(T) * S_0)
    V_0 = S_0 * (ss.norm.cdf(d1) + sigma * np.sqrt(T) * ss.norm.pdf(d1))
    return V_0


def Bachelier_daon_put(S_0, K, T, sigma):
    '''digital asset-or-nothing put option pricing fomula under Bachelier Model'''
    d1 = (S_0 - K) / (sigma * np.sqrt(T) * S_0)
    V_0 = S_0 * (ss.norm.cdf(-d1) - sigma * np.sqrt(T) * ss.norm.pdf(-d1))
    return V_0


#Black76 Model ps.d1,d2 in all functions come from call option
#1 vanilla option
def black76_call(F_0, K, T, r, sigma):
    '''European call option pricing fomula under black76 model'''
    d1 = (np.log(F_0 / K) + 0.5 * T * sigma**2) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    V_0 = np.exp(-r * T) * (F_0 * ss.norm.cdf(d1) - K * ss.norm.cdf(d2))
    return V_0


def black76_put(F_0, K, T, r, sigma):
    '''European put option pricing fomula under black76 model'''
    d1 = (np.log(F_0 / K) + 0.5 * T * sigma**2) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    V_0 = np.exp(-r * T) * (-F_0 * ss.norm.cdf(-d1) + K * ss.norm.cdf(-d2))
    return V_0


#2 digital cash or nothing
def black76_dcon_call(F_0, K, T, r, sigma):
    '''digital cash-or-nothing call option pricing fomula under black76 model'''
    d1 = (np.log(F_0 / K) + 0.5 * T * sigma**2) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    V_0 = np.exp(-r * T) * ss.norm.cdf(d2)
    return V_0


def black76_dcon_put(F_0, K, T, r, sigma):
    '''digital cash-or-nothing put option pricing fomula under black76 model'''
    d1 = (np.log(F_0 / K) + 0.5 * T * sigma**2) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    V_0 = np.exp(-r * T) * ss.norm.cdf(-d2)
    return V_0


#3 digital asset or nothing
def black76_daon_call(F_0, K, T, r, sigma):
    '''digital asset-or-nothing call option pricing fomula under black76 model'''
    d1 = (np.log(F_0 / K) + 0.5 * T * sigma**2) / (sigma * np.sqrt(T))
    V_0 = np.exp(-r * T) * ss.norm.cdf(d1)
    return V_0


def black76_daon_put(F_0, K, T, r, sigma):
    '''digital asset-or-nothing put option pricing fomula under black76 model'''
    d1 = (np.log(F_0 / K) + 0.5 * T * sigma**2) / (sigma * np.sqrt(T))
    V_0 = np.exp(-r * T) * ss.norm.cdf(-d1)
    return V_0


#Displaced-Diffusion Model ps.d1,d2 in all functions come from call option
#1 vanilla option
def displaced_diffusion_call(F, K, T, r, sigma, beta):
    '''European call option pricing fomula under Displaced-Diffusion Model'''
    F_0 = F / beta
    K = ((1 - beta) / beta) * F + K
    sigma = sigma * beta
    d1 = (np.log(F_0 / K) + 0.5 * T * sigma**2) / (sigma * np.sqrt(T))
    d2 = (np.log(F_0 / K) - 0.5 * T * sigma**2) / (sigma * np.sqrt(T))
    V_0 = np.exp(-r * T) * (F_0 * ss.norm.cdf(d1) - K * ss.norm.cdf(d2))
    return V_0


def displaced_diffusion_put(F, K, T, r, sigma, beta):
    '''European put option pricing fomula under Displaced-Diffusion Model'''
    F_0 = F / beta
    K = ((1 - beta) / beta) * F + K
    sigma_n = sigma * beta
    d1 = (np.log(F_0 / K) + 0.5 * T * sigma_n**2) / (sigma_n * np.sqrt(T))
    d2 = d1 - sigma_n * np.sqrt(T)
    V_0 = np.exp(-r * T) * (-F_0 * ss.norm.cdf(-d1) + K * ss.norm.cdf(-d2))
    return V_0


#2 digital cash or nothing
def displaced_diffusion_dcon_call(F, K, T, r, sigma, beta):
    '''digital cash-or-nothing call option pricing fomula under Displaced-Diffusion Model'''
    F_0 = F / beta
    K = ((1 - beta) / beta) * F + K
    sigma = sigma * beta
    d1 = (np.log(F_0 / K) + 0.5 * T * sigma**2) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    V_0 = np.exp(-r * T) * ss.norm.cdf(d2)
    return V_0


def displaced_diffusion_dcon_put(F, K, T, r, sigma, beta):
    '''digital cash-or-nothing put option pricing fomula under Displaced-Diffusion Model'''
    F_0 = F / beta
    K = ((1 - beta) / beta) * F + K
    sigma = sigma * beta
    d1 = (np.log(F_0 / K) + 0.5 * T * sigma**2) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    V_0 = np.exp(-r * T) * ss.norm.cdf(-d2)
    return V_0


#3 digital asset or nothing
def displaced_diffusion_daon_call(F, K, T, r, sigma, beta):
    '''digital asset-or-nothing call option pricing fomula under Displaced-Diffusion Model'''
    F_0 = F / beta
    K = ((1 - beta) / beta) * F + K
    sigma = sigma * beta
    d1 = (np.log(F_0 / K) + 0.5 * T * sigma**2) / (sigma * np.sqrt(T))
    V_0 = F_0 * np.exp(-r * T) * ss.norm.cdf(d1)
    return V_0


def displaced_diffusion_daon_put(F, K, T, r, sigma, beta):
    '''digital asset-or-nothing put option pricing fomula under Displaced-Diffusion Model'''
    F_0 = F / beta
    K = ((1 - beta) / beta) * F + K
    sigma = sigma * beta
    d1 = (np.log(F_0 / K) + 0.5 * T * sigma**2) / (sigma * np.sqrt(T))
    V_0 = F_0 * np.exp(-r * T) * ss.norm.cdf(-d1)
    return V_0
