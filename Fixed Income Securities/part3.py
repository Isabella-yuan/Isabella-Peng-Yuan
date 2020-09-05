# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.optimize import brentq
from scipy.interpolate import interp2d
from scipy.misc import derivative
from scipy.integrate import quad
from scipy.stats import norm

OIS_sheet = pd.read_excel('IR Data.xlsx', sheet_name='OIS', usecols='A:C')
tenor = np.array([[float(i[:-1])] for i in OIS_sheet.Tenor.values])
tenor[0] = [0.5]
rate = OIS_sheet.Rate.values.reshape(-1, 1)
OIS_discount = np.array([
    0.9987515605493134, 0.9970089730807579, 0.9935307459132694,
    0.9900151412182886, 0.9861166497152535, 0.9821841197332123,
    0.9724057745943246, 0.9559768785262047, 0.9276114796053301,
    0.9000759370413394, 0.8474067157362282
])
x_t = np.append(0, tenor.reshape(-1))
y_t = np.append(1, OIS_discount.reshape(-1))
ODF_function = interpolate.interp1d(x_t, y_t)
ODF_semi_array = ODF_function(np.arange(0.5, 30 + 0.5, 0.5))
ODF_quarterly_array = ODF_function(np.arange(0.25, 30 + 0.25, 0.25))


def D_to_L(D_array, t):
    L_array = (D_array[0: int(2*t)]-D_array[1: int(2*t)+1]) / \
        D_array[1: int(2*t)+1]*2
    return L_array


def fix_flt_solve(x, t, t_lag_1):
    fix = IRS_dict[int(t)] * ODF_semi_array[0:int(t * 2)].sum()
    LDF_semi_array[int(t_lag_1 * 2):int(t * 2 + 1)] = np.linspace(
        LDF_semi_array[int(t_lag_1 * 2)], x, int((t - t_lag_1) * 2 + 1))
    flt = np.sum(ODF_semi_array[0:int(t * 2)] * D_to_L(LDF_semi_array, t))
    return fix - flt


IRS_sheet = pd.read_excel('IR Data.xlsx', sheet_name='IRS', usecols='A:C')
IRS_dict = dict(
    zip(list(tenor[:].reshape(-1)),
        IRS_sheet.Rate.values.reshape(-1).tolist()))
LDF_semi_array = np.zeros(61)
LDF_semi_array[0] = 1
LDF_semi_array[1] = 1 / (1 + 0.025 / 2)
time_range = tenor.reshape(-1)
n = 1
for n in range(1, len(time_range)):
    t = time_range[n]
    t_lag_1 = time_range[n - 1]
    discount_T = brentq(lambda x: fix_flt_solve(x, t, t_lag_1), 0.2, 1)
    LDF_semi_array[int(t_lag_1 * 2):int(t * 2 + 1)] = np.linspace(
        LDF_semi_array[int(t_lag_1 * 2)], discount_T, (t - t_lag_1) * 2 + 1)
Libor_semi_array = D_to_L(LDF_semi_array, 30)

swap_sheet = pd.read_excel('IR Data.xlsx',
                           sheet_name='Swaption',
                           skiprows=2,
                           usecols='A:B')

Expiry = np.array([[float(i[:-1])]
                   for i in swap_sheet.Expiry.values]).reshape(-1)
Tenor = np.array([[float(i[:-1])]
                  for i in swap_sheet.Tenor.values]).reshape(-1)

E_T_semi = np.c_[np.array([[0.5]] * 10).cumsum(axis=0),
                 np.array([[10]] * 10) + np.array([[0.5]] * 10).cumsum(axis=0)]
Forward_semi_swap=np.array([ sum(ODF_semi_array[int(i*2):int((i+j)*2)]*Libor_semi_array[int(i*2):int((i+j)*2)])\
   /sum(ODF_semi_array[int(i*2):int((i+j)*2)]) for i,j in E_T_semi])
Forward_semi_swap

alpha, rho, nu = (np.array([[0.146945, 0.189750, 0.203499, 0.185444, 0.179290],
                            [0.163509, 0.199079, 0.211619, 0.194700, 0.178872],
                            [0.173957, 0.191431, 0.02754, 0.201465,
                             0.194105]]),
                  np.array(
                      [[-0.608186, -0.525113, -0.495981, -0.455127, -0.350125],
                       [-0.564171, -0.540392, -0.550340, -0.528894, -0.458390],
                       [-0.530788, -0.531739, -0.537957, -0.561475,
                        -0.569189]]),
                  np.array([[1.931027, 1.624788, 1.378179, 0.987900, 0.671529],
                            [1.304129, 1.050802, 0.926734, 0.653658, 0.485775],
                            [0.989753, 0.912260, 0.856719, 0.717186,
                             0.554851]]))


def interpolation_greek(e, t, greek):
    a = np.array([1, 2, 3, 5, 10])
    b = np.array([1, 5, 10])
    f = interp2d(a, b, greek)
    return f(e, t)


def CMS_func(S, delta, T, tenor, alpha, beta, rho, nu):
    def IRR_function(S):
        temp = sum(delta / (1 + delta * S)**i
                   for i in range(1, int(tenor / delta + 1)))
        return temp

    def BK76_payer(K):
        sigma = min(
            0.7,
            SABR(S, K, T, interpolation_greek(T, tenor, alpha), beta,
                 interpolation_greek(T, tenor, rho),
                 interpolation_greek(T, tenor, nu)))
        d1 = (np.log(S / K) + 1 / 2 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * norm.cdf(d2)

    def BK76_receiver(K):
        sigma = min(
            0.7,
            SABR(S, K, T, interpolation_greek(T, tenor, alpha), beta,
                 interpolation_greek(T, tenor, rho),
                 interpolation_greek(T, tenor, nu)))
        d1 = (np.log(S / K) + 1 / 2 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * norm.cdf(-d2) - S * norm.cdf(-d1)

    def H_prime2(K):
        IRR_prime = derivative(IRR_function, K, dx=0.0001)
        IRR_prime2 = derivative(IRR_function, K, dx=0.0001, n=2)
        h = (-IRR_prime2*K-2*IRR_prime)/IRR_function(K)**2\
            + (2*IRR_prime**2*K)/IRR_function(K)**3
        return h

    cms = S+IRR_function(S)*quad(lambda x:
                                 H_prime2(x)*BK76_receiver(x), 0, S)[0]\
        + IRR_function(S)*quad(lambda x:
                               H_prime2(x)*BK76_payer(x), S, np.Inf)[0]
    return cms


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


CMS_semi = [
    CMS_func(Forward_semi_swap[i], 1 / 2, E_T_semi[i, 0], E_T_semi[i, 1],
             alpha, 0.9, rho, nu) for i in range(len(Forward_semi_swap))
]
PV_semi = np.sum([
    0.5 * ODF_semi_array[i] * CMS_semi[i]
    for i in range(len(Forward_semi_swap))
])


def D_to_L_quarterly(D_array, t):
    L_array = (D_array[0: int(4*t)]-D_array[1: int(4*t)+1]) / \
        D_array[1: int(4*t)+1]*4
    return L_array


def fix_flt_solve_quarterly(x, t, t_lag_1):
    fix = IRS_dict[int(t)] * ODF_quarterly_array[0:int(t * 4)].sum()
    LDF_quarterly_array[int(t_lag_1 * 4):int(t * 4 + 1)] = np.linspace(
        LDF_quarterly_array[int(t_lag_1 * 4)], x, int((t - t_lag_1) * 4 + 1))
    flt = np.sum(ODF_quarterly_array[0:int(t * 4)] *
                 D_to_L_quarterly(LDF_quarterly_array, t))
    return fix - flt


LDF_quarterly_array = np.zeros(121)
LDF_quarterly_array[0] = 1
LDF_quarterly_array[2] = 1 / (1 + 0.025 / 2)
LDF_quarterly_array[1] = (LDF_quarterly_array[0] + LDF_quarterly_array[2]) / 2

for n in range(1, len(time_range)):
    t = time_range[n]
    t_lag_1 = time_range[n - 1]
    discount_T = brentq(lambda x: fix_flt_solve_quarterly(x, t, t_lag_1), 0.2,
                        1)
    LDF_quarterly_array[int(t_lag_1 * 4):int(t * 4 + 1)] = np.linspace(
        LDF_quarterly_array[int(t_lag_1 * 4)], discount_T,
        (t - t_lag_1) * 4 + 1)
Libor_quarterly_array = D_to_L_quarterly(LDF_quarterly_array, 30)

E_T_quarter = np.c_[np.array([[0.25]] * 40).cumsum(axis=0),
                    np.array([[2]] * 40) +
                    np.array([[0.25]] * 40).cumsum(axis=0)]
Forward_quarter_swap=np.array([ sum(ODF_quarterly_array[int(i*4):int((i+j)*4)]*Libor_quarterly_array[int(i*4):int((i+j)*4)])\
/sum(ODF_quarterly_array[int(i*4):int((i+j)*4)]) for i,j in E_T_quarter])

#calculating cms
CMS_quarter = [
    CMS_func(Forward_quarter_swap[i], 1 / 2, E_T_quarter[i, 0],
             E_T_quarter[i, 1], alpha, 0.9, rho, nu)
    for i in range(len(Forward_quarter_swap))
]
PV_quarter = (np.sum([
    0.5 * ODF_quarterly_array[i] * CMS_quarter[i]
    for i in range(len(Forward_quarter_swap))
])) / 20

Forward_swap = np.array([
    0.0320069991211797, 0.033259225750316514, 0.034010726324991546,
    0.035255471532617884, 0.03842702, 0.039274036967966754,
    0.040074861093919396, 0.04007224645790493, 0.04109322876656141,
    0.04363364552747551, 0.0421892371404898, 0.043115736402897675,
    0.044097124988326346, 0.04624901920952453, 0.05345752914372313
])
CMS_ET = [
    CMS_func(Forward_swap[i], 1 / 2,
             Tenor.reshape(-1)[i],
             Expiry.reshape(-1)[i], alpha, 0.9, rho, nu)
    for i in range(len(Forward_swap))
]

print(PV_semi, CMS_semi, CMS_quarter, PV_quarter, CMS_ET)