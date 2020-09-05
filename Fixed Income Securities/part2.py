import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, least_squares
from enum import Enum
import part1 as fixed_income
from math import log

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 100)


class frequency(Enum):
    quarter = 0.25
    semi_annual = 0.5
    annual = 1


def generate_pv01(expiry, tenor, daycount):
    pv01 = 0
    for i in range(int(expiry / daycount + 1),
                   int((expiry + tenor) / daycount + 1)):
        pv01 += daycount * OIS_instance.DF(0, i * daycount)
    return pv01


#  Displaced-Diffusion
def dd_swaption_payer(F0, K, n, N, beta, dd_sigma):
    pv01 = df_pv01.loc[(n, N), 'PVBP']
    d1 = (np.log(F0 / (beta * (K + (1 - beta) / beta * F0))) + 0.5 * n *
          (beta * dd_sigma)**2) / (beta * dd_sigma * np.sqrt(n))
    d2 = d1 - (dd_sigma * beta) * np.sqrt(n)
    return pv01 * ((F0 / beta) * norm.cdf(d1) -
                   (K + (1 - beta) / beta * F0) * norm.cdf(d2))


def dd_swaption_receiver(F0, K, n, N, dd_beta, sigma):
    pv01 = df_pv01.loc[(n, N), 'PVBP']
    d1 = (np.log(F0 / (dd_beta * (K +
                                  (1 - dd_beta) / dd_beta * F0))) + 0.5 * n *
          (dd_beta * sigma)**2) / (dd_beta * sigma * np.sqrt(n))
    d2 = d1 - (sigma * dd_beta) * np.sqrt(n)
    return pv01 * ((K + ((1 - dd_beta) / dd_beta) * F0) * norm.cdf(-d2) -
                   (F0 / dd_beta) * norm.cdf(-d1))


def swaption_payer(S, K, sigma, n, N):  # n is the expiry of the option
    pv01 = df_pv01.loc[(n, N), 'PVBP']
    d1 = (np.log(S / K) + 0.5 * sigma**2 * n) / (sigma * np.sqrt(n))
    d2 = d1 - sigma * np.sqrt(n)
    return pv01 * (S * norm.cdf(d1) - K * norm.cdf(d2))


def swaption_receiver(S, K, sigma, n, N):  # n is the expiry of the option
    pv01 = df_pv01.loc[(n, N), 'PVBP']
    d1 = (np.log(S / K) + 0.5 * sigma**2 * n) / (sigma * np.sqrt(n))
    d2 = d1 - sigma * np.sqrt(n)
    return pv01 * (K * norm.cdf(-d2) - S * norm.cdf(-d1))


def swaption_prices(S, K, sigma, n, N):
    if K < S:
        return swaption_receiver(S, K, sigma, n, N)
    else:
        return swaption_payer(S, K, sigma, n, N)


def implied_Vol(S, K, n, N, price):
    if K < S:
        return brentq(lambda x: price - swaption_receiver(S, K, x, n, N), 1e-6,
                      1)
    else:
        return brentq(lambda x: price - swaption_payer(S, K, x, n, N), 1e-6, 1)


def dd_model(S, K, n, N, beta, dd_sigma):
    if K < S:
        dd_price = dd_swaption_receiver(S, K, n, N, beta, dd_sigma)
    else:
        dd_price = dd_swaption_payer(S, K, n, N, beta, dd_sigma)
    return dd_price


def dd_calibration_price(
        x, swap, n, N, strikes, prices
):  # beta & sigma guesses in x; list of strikes & respective market vols
    err = 0.0
    for i, price in enumerate(prices):
        dd_price = dd_model(swap, strikes[i], n, N, x[0], x[1])
        err += (price - dd_price)**2
    return err


data_path = 'IR Data.xlsx'
OIS_instance = fixed_income.OIS(frequency.semi_annual)
IRS_instance = fixed_income.IRS(frequency.semi_annual, OIS_instance.DF)
fixed_income.data_processor(OIS_instance, data_path, 'OIS')
fixed_income.data_processor(IRS_instance, data_path, 'IRS')

df_vol = pd.read_excel(data_path, sheet_name="Swaption", header=2)
df_vol['Expiry'] = df_vol['Expiry'].map(lambda x: int(x.rstrip('Y')))
df_vol['Tenor'] = df_vol['Tenor'].map(lambda x: int(x.rstrip('Y')))
df_vol = df_vol.set_index(['Expiry', 'Tenor'])
df_vol = df_vol.apply(lambda x: x / 100)
# print(df_vol)

# construct DataFrame of strikes
strike_diff = [[
    -0.02, -0.015, -0.01, -0.005, -0.0025, 0, 0.0025, 0.005, 0.01, 0.015, 0.02
]] * len(df_vol.index)
df_strikes = pd.DataFrame(data=np.asarray(strike_diff),
                          index=df_vol.index,
                          columns=df_vol.columns)

# construct DataFrame of DD parameters
df_dd_parameters = pd.DataFrame(np.full((len(df_vol.index), 2), np.nan),
                                index=df_vol.index,
                                columns=["Beta", "Sigma"])

# construct DataFrame of SABR parameters
df_SABR_parameters = pd.DataFrame(np.full((len(df_vol.index), 3), np.nan),
                                  index=df_vol.index,
                                  columns=["Alpha", "Rho", "Nu"])

# construct DataFrame of swaption prices
df_swaption_prices = pd.DataFrame(np.full(
    (len(df_vol.index), len(strike_diff[0])), np.nan),
                                  index=df_vol.index,
                                  columns=df_vol.columns)

# *****************************  construct DataFrame of PV01 *****************************
df_pv01 = pd.DataFrame(np.full((len(df_vol.index), 1), np.nan),
                       index=df_vol.index,
                       columns=['PVBP'])
for i in df_pv01.index:
    expiry, tenor = i
    daycount = 0.5  # frequency.value
    df_pv01.loc[(expiry,
                 tenor), 'PVBP'] = generate_pv01(expiry, tenor, daycount)

# loop for calibrating Displaced-Diffusion model parameters for all maturity and tenor pairs
for i in range(len(df_strikes.index)):
    expiry = df_strikes.index.get_level_values(0)[i]
    tenor = df_strikes.index.get_level_values(1)[i]
    swap = IRS_instance.IRS_pricer(expiry, tenor)
    df_strikes.loc[(expiry,
                    tenor)] = df_strikes.loc[(expiry,
                                              tenor)].apply(lambda x: x + swap)

    for j in df_swaption_prices.columns:
        df_swaption_prices.loc[(expiry, tenor), j] = swaption_prices(
            swap, df_strikes.loc[(expiry, tenor), j],
            df_vol.loc[(expiry, tenor), j], expiry, tenor)

    strikes = df_strikes.loc[(expiry, tenor)].values
    vols = df_vol.loc[(expiry, tenor)].values
    prices = df_swaption_prices.loc[(expiry, tenor)].values

    initialGuess = [0.5, 0.5]
    res = least_squares(lambda x: dd_calibration_price(x, swap, expiry, tenor,
                                                       strikes, prices),
                        initialGuess,
                        bounds=(1e-6, 1))
    dd_beta, dd_sigma = res.x[0], res.x[1]
    df_dd_parameters.loc[(expiry, tenor), "Beta"] = dd_beta
    df_dd_parameters.loc[(expiry, tenor), "Sigma"] = dd_sigma


#  SABR model
def SABR(F, K, T, alpha, beta, rho, nu):
    X = K
    # if K is at-the-money-forward
    if abs(F - K) < 1e-12:
        numer1 = (((1 - beta)**2) / 24) * alpha * alpha / (F**(2 - 2 * beta))
        numer2 = 0.25 * rho * beta * nu * alpha / (F**(1 - beta))
        numer3 = ((2 - 3 * rho * rho) / 24) * nu * nu
        VolAtm = alpha * (1 + (numer1 + numer2 + numer3) * T) / (F**(1 - beta))
        sabrsigma = VolAtm
    else:
        z = (nu / alpha) * ((F * X)**(0.5 * (1 - beta))) * log(F / X)
        zhi = log((((1 - 2 * rho * z + z * z)**0.5) + z - rho) / (1 - rho))
        numer1 = (((1 - beta)**2) / 24) * ((alpha * alpha) /
                                           ((F * X)**(1 - beta)))
        numer2 = 0.25 * rho * beta * nu * alpha / ((F * X)**((1 - beta) / 2))
        numer3 = ((2 - 3 * rho * rho) / 24) * nu * nu
        numer = alpha * (1 + (numer1 + numer2 + numer3) * T) * z
        denom1 = ((1 - beta)**2 / 24) * (log(F / X))**2
        denom2 = (((1 - beta)**4) / 1920) * ((log(F / X))**4)
        denom = ((F * X)**((1 - beta) / 2)) * (1 + denom1 + denom2) * zhi
        sabrsigma = numer / denom

    return sabrsigma


#  SABR calibration
beta = 0.9


def sabrcalibration(x, strikes, vols, F, T):
    err = 0.0
    for i, vol in enumerate(vols):
        err += (vol - SABR(F, strikes[i], T, x[0], beta, x[1], x[2]))**2
    return err


# reinitialize strike DataFrame (needed because design was to run either DD or SABR, so if both, DF needs to be reset)
df_strikes = pd.DataFrame(data=np.asarray(strike_diff),
                          index=df_vol.index,
                          columns=df_vol.columns)

# lopp to calibrate SABR parameters for each maturity and tenor pairs
for i in range(len(
        df_strikes.index)):  # change range to len(df_strikes.index) later
    expiry = df_strikes.index.get_level_values(0)[i]
    tenor = df_strikes.index.get_level_values(1)[i]
    swap = IRS_instance.IRS_pricer(expiry, tenor)
    df_strikes.loc[(expiry,
                    tenor)] = df_strikes.loc[(expiry,
                                              tenor)].apply(lambda x: x + swap)

    for j in df_swaption_prices.columns:
        df_swaption_prices.loc[(expiry, tenor), j] = swaption_prices(
            swap, df_strikes.loc[(expiry, tenor), j],
            df_vol.loc[(expiry, tenor), j], expiry, tenor)

    strikes = df_strikes.loc[(expiry, tenor)].values
    vols = df_vol.loc[(expiry, tenor)].values
    prices = df_swaption_prices.loc[(expiry, tenor)].values

    initialGuess = [0.9, -0.2, 0.1]
    res = least_squares(
        lambda x: sabrcalibration(x, strikes, vols, swap, expiry),
        initialGuess)
    alpha = res.x[0]
    rho = res.x[1]
    nu = res.x[2]

    df_SABR_parameters.loc[(expiry, tenor), "Alpha"] = alpha
    df_SABR_parameters.loc[(expiry, tenor), "Rho"] = rho
    df_SABR_parameters.loc[(expiry, tenor), "Nu"] = nu

# Computing Part 2.3 answers: pricing of swaptions that require interpolation of DD and SABR parameters
# interpolating DD and SABR parameters
df_dd_parameters.loc[(2, 10), "Beta"] = (df_dd_parameters.loc[(5, 10), "Beta"]
                                         - df_dd_parameters.loc[(1, 10), "Beta"]) / 5\
                                        + df_dd_parameters.loc[(1, 10), "Beta"]
df_dd_parameters.loc[(2, 10), "Sigma"] = (df_dd_parameters.loc[(5, 10), "Sigma"]
                                          - df_dd_parameters.loc[(1, 10), "Sigma"]) / 5\
                                         + df_dd_parameters.loc[(1, 10), "Sigma"]
df_dd_parameters.loc[(8, 10), "Beta"] = (df_dd_parameters.loc[(10, 10), "Beta"]
                                         - df_dd_parameters.loc[(5, 10), "Beta"]) / 5\
                                        + df_dd_parameters.loc[(5, 10), "Beta"]
df_dd_parameters.loc[(8, 10), "Sigma"] = (df_dd_parameters.loc[(10, 10), "Sigma"]
                                          - df_dd_parameters.loc[(5, 10), "Sigma"]) / 5\
                                         + df_dd_parameters.loc[(5, 10), "Sigma"]

df_SABR_parameters.loc[(2,10),"Alpha"] = (df_SABR_parameters.loc[(5,10),"Alpha"]
                                          - df_SABR_parameters.loc[(1,10),"Alpha"]) / 5\
                                         + df_SABR_parameters.loc[(1,10),"Alpha"]
df_SABR_parameters.loc[(2,10),"Rho"] = (df_SABR_parameters.loc[(5,10),"Rho"]
                                        - df_SABR_parameters.loc[(1,10),"Rho"]) / 5\
                                       + df_SABR_parameters.loc[(1,10),"Rho"]
df_SABR_parameters.loc[(2,10),"Nu"] = (df_SABR_parameters.loc[(5,10),"Nu"]
                                       - df_SABR_parameters.loc[(1,10),"Nu"]) / 5\
                                      + df_SABR_parameters.loc[(1,10),"Nu"]

df_SABR_parameters.loc[(8,10),"Alpha"] = (df_SABR_parameters.loc[(10,10),"Alpha"]
                                          - df_SABR_parameters.loc[(5,10),"Alpha"]) / 5\
                                         + df_SABR_parameters.loc[(5,10),"Alpha"]
df_SABR_parameters.loc[(8,10),"Rho"] = (df_SABR_parameters.loc[(10,10),"Rho"]
                                        - df_SABR_parameters.loc[(5,10),"Rho"]) / 5\
                                       + df_SABR_parameters.loc[(5,10),"Rho"]
df_SABR_parameters.loc[(8,10),"Nu"] = (df_SABR_parameters.loc[(10,10),"Nu"]
                                       - df_SABR_parameters.loc[(5,10),"Nu"]) / 5\
                                      + df_SABR_parameters.loc[(5,10),"Nu"]

daycount = 0.5
df_pv01.loc[(2, 10), 'PVBP'] = generate_pv01(2, 10, daycount)
df_pv01.loc[(8, 10), 'PVBP'] = generate_pv01(8, 10, daycount)

# create DataFrame to store swaption prices computed using DD and SABR parameters
df_dd_prices = pd.DataFrame(
    data=[],
    index=pd.MultiIndex.from_tuples([(2, 10)], names=['Expiry', 'Tenor']),
    columns=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
df_sabr_prices = pd.DataFrame(
    data=[],
    index=pd.MultiIndex.from_tuples([(2, 10)], names=['Expiry', 'Tenor']),
    columns=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])

# loop to compute prices
for k in df_dd_prices.columns:
    swap = IRS_instance.IRS_pricer(2, 10)
    df_dd_prices.loc[(2, 10), k] = dd_swaption_payer(
        swap, k, 2, 10, df_dd_parameters.loc[(2, 10), "Beta"],
        df_dd_parameters.loc[(2, 10), "Sigma"])
    df_sabr_prices.loc[(2, 10), k] = swaption_payer(
        swap, k,
        SABR(swap, k, 2, df_SABR_parameters.loc[(2, 10), "Alpha"], 0.9,
             df_SABR_parameters.loc[(2, 10), "Rho"],
             df_SABR_parameters.loc[(2, 10), "Nu"]), 2, 10)

    swap = IRS_instance.IRS_pricer(8, 10)
    df_dd_prices.loc[(8, 10), k] = dd_swaption_receiver(
        swap, k, 8, 10, df_dd_parameters.loc[(8, 10), "Beta"],
        df_dd_parameters.loc[(8, 10), "Sigma"])
    df_sabr_prices.loc[(8, 10), k] = swaption_receiver(
        swap, k,
        SABR(swap, k, 8, df_SABR_parameters.loc[(8, 10), "Alpha"], 0.9,
             df_SABR_parameters.loc[(8, 10), "Rho"],
             df_SABR_parameters.loc[(8, 10), "Nu"]), 8, 10)

df_pv01 = df_pv01.sort_index()
df_dd_parameters = df_dd_parameters.sort_index()
df_SABR_parameters = df_SABR_parameters.sort_index()
print("\nTable of PV01 values:\n", df_pv01)
print("\nTable of DD Parameters:\n", df_dd_parameters)
print("\nTable of SABR parameters\n", df_SABR_parameters)
print("\nTable of swaption prices computed under DD model:\n", df_dd_prices)
print("\nTable of swaption prices computed under SABR model:\n",
      df_sabr_prices)
df_pv01.to_excel('df_pv01.xlsx')
df_dd_parameters.to_excel('df_dd_parameters.xlsx')
df_SABR_parameters.to_excel('df_SABR_parameters.xlsx')
df_dd_prices.to_excel('df_dd_prices.xlsx')
df_sabr_prices.to_excel('df_sabr_prices.xlsx')
