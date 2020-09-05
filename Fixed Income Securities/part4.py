import part2 as F2
import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.optimize import brentq
from scipy.interpolate import interp2d
from scipy.misc import derivative
from scipy.integrate import quad
from scipy.stats import norm


def SABR(F, K, T, alpha, beta, rho, nu):
    X = K
    if F == K:
        numer1 = (((1 - beta)**2) / 24) * alpha * alpha / (F**(2 - 2 * beta))
        numer2 = 0.25 * rho * beta * nu * alpha / (F**(1 - beta))
        numer3 = ((2 - 3 * rho * rho) / 24) * nu * nu
        VolAtm = alpha * (1 + (numer1 + numer2 + numer3) * T) / (F**(1 - beta))
        sabrsigma = VolAtm
    else:
        z = (nu / alpha) * ((F * X)**(0.5 * (1 - beta))) * np.log(F / X)
        zhi = np.log((((1 - 2 * rho * z + z * z)**0.5) + z - rho) / (1 - rho))
        numer1 = (((1 - beta)**2) / 24) * ((alpha * alpha) /
                                           ((F * X)**(1 - beta)))
        numer2 = 0.25 * rho * beta * nu * alpha / ((F * X)**((1 - beta) / 2))
        numer3 = ((2 - 3 * rho * rho) / 24) * nu * nu
        numer = alpha * (1 + (numer1 + numer2 + numer3) * T) * z
        denom1 = ((1 - beta)**2 / 24) * (np.log(F / X))**2
        denom2 = (((1 - beta)**4) / 1920) * ((np.log(F / X))**4)
        denom = ((F * X)**((1 - beta) / 2)) * (1 + denom1 + denom2) * zhi
        sabrsigma = numer / denom
    return sabrsigma


def decomp_Option(S, delta, T, tenor, alpha, beta, rho, nu):
    def IRR_function(S):
        temp = sum(delta / (1 + delta * S)**i
                   for i in range(1, int(tenor / delta + 1)))
        return temp

    def BK76_payer(K):
        sigma = min(0.7, SABR(S, K, T, alpha, beta, rho, nu))
        d1 = (np.log(S / K) + 1 / 2 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * norm.cdf(d2)

    def BK76_receiver(K):
        sigma = min(0.7, SABR(S, K, T, alpha, beta, rho, nu))
        d1 = (np.log(S / K) + 1 / 2 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * norm.cdf(-d2) - S * norm.cdf(-d1)

    def IRR1d(K):
        return derivative(IRR_function, K, dx=0.0001)

    def IRR2d(K):
        return derivative(IRR_function, K, dx=0.0001, n=2)

    def g(K):
        return K**(1 / 4) - 0.04**(1 / 2)

    def g1d(K):
        return derivative(g, K, dx=0.0001)

    def g2d(K):
        return derivative(g, K, dx=0.0001, n=2)

    def H_prime2(K):
        h_prime2 = (IRR_function(K)*g2d(K)-IRR2d(K)*g(K)-2*IRR1d(K)*g1d(K))/IRR_function(K)**2 +\
            2*g(K)*IRR1d(K)**2/IRR_function(K)**3
        return h_prime2

    def H_prime(K):
        h_prime1 = (IRR_function(K) * g1d(K) -
                    g(K) * IRR1d(K)) / IRR_function(K)**2
        return h_prime1

    pv1 = g(S)*F2.OIS_instance.DF(0,5) + (quad(lambda x: H_prime2(x)*BK76_receiver(x), 0, S)[0] + \
        quad(lambda x: H_prime2(x)*BK76_payer(x), S, np.Inf)[0]) * F2.OIS_instance.DF(0,5) * IRR_function(S)

    pv2 = H_prime(0.2**4) * F2.OIS_instance.DF(
        0, 5) * IRR_function(S) * BK76_payer(0.2**4) + F2.OIS_instance.DF(
            0, 5) * IRR_function(S) * quad(
                lambda x: H_prime2(x) * BK76_payer(x), 0.2**4, np.Inf)[0]

    return pv1, pv2


if __name__ == "__main__":
    par = [0.17862487, 0.9, -0.431018952, 0.46325639]
    S_5_15 = 0.04363364552747551
    pv1, pv2 = decomp_Option(S_5_15, 0.5, 5, 10, par[0], par[1], par[2],
                             par[3])[:]
    print(pv1, pv2)
