import part1 as op
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, least_squares
from scipy.interpolate import interp1d
from datetime import datetime


def impliedCallVolatility(S_0, K, T, r, price):
    implied_Vol = brentq(
        lambda x: price - op.Black_Scholes_call(S_0, K, T, r, x), 1e-6, 1)
    return implied_Vol


def impliedPutVolatility(S_0, K, T, r, price):
    implied_Vol = brentq(
        lambda x: price - op.Black_Scholes_put(S_0, K, T, r, x), 1e-6, 1)
    return implied_Vol


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


def r_interpolate(file_name, T):
    import pandas as pd
    from scipy.interpolate import interp1d
    df_rate = pd.read_csv(file_name)
    x = df_rate.iloc[:, 0].values
    y = df_rate.iloc[:, 1].values
    f = interp1d(x, y)
    return f(T) / 100


#--------------------------------------------------------------
df_goog_call = pd.read_csv('goog_call.csv', index_col=0, parse_dates=True)
df_goog_put = pd.read_csv('goog_put.csv', index_col=0, parse_dates=True)
df_goog_price = pd.read_csv('GOOG.csv', index_col=0, parse_dates=True)

S_0 = df_goog_price['close'][df_goog_price.index ==
                             df_goog_call.index[0]].values[0]

df_dicount_rate = pd.read_csv('discount.csv')

T_0 = pd.Timestamp(
    datetime.strptime(str(df_goog_call['expiry'][0]), '%Y%m%d').date())
T = (T_0 - df_goog_call.index[0]).days / 365

r = r_interpolate('discount.csv', T * 365)

F_0 = S_0 * np.exp(r * T)

df_goog_call = df_goog_call[
    df_goog_call['strike'] >
    F_0]  # slicing smile right side ATM = S_0 * np.exp(r * T)
df_goog_put = df_goog_put[df_goog_put['strike'] <
                          F_0]  # slicing smile left side

df_goog_call['price'] = (df_goog_call['best_bid'] +
                         df_goog_call['best_offer']) / 2
df_goog_put['price'] = (df_goog_put['best_bid'] +
                        df_goog_put['best_offer']) / 2

df_goog_call['market_iv'] = df_goog_call.apply(
    lambda x: impliedCallVolatility(S_0, x['strike'], T, r, x['price']),
    axis=1)
df_call = df_goog_call[['strike', 'price', 'market_iv']]
df_goog_put['market_iv'] = df_goog_put.apply(
    lambda x: impliedPutVolatility(S_0, x['strike'], T, r, x['price']), axis=1)
df_put = df_goog_put[['strike', 'price', 'market_iv']]

figure1 = plt.figure(figsize=(10, 5))
ax1 = figure1.add_subplot()
ax1.plot(df_call['strike'],
         df_call['market_iv'],
         'o',
         label='call option',
         markersize=2)
ax1.plot(df_put['strike'],
         df_put['market_iv'],
         'o',
         label='put option',
         markersize=2)
ax1.grid(alpha=0.2)
ax1.set_xlabel('strike')
ax1.set_ylabel('implied vol')

df_smile_summary = df_put.append(df_call,
                                 ignore_index=True)[['strike', 'market_iv']]
df_smile_summary['diff'] = df_smile_summary['strike'].apply(
    lambda x: np.abs(x - F_0))
ATM_iv = df_smile_summary.sort_values(by='diff')['market_iv'].values[0]

#-------------------------------displaced diffusion model calibration--------------------------------


def error(x, strikes, vols):
    err = 0.0
    for i, vol in enumerate(vols):
        price = op.displaced_diffusion_call(S_0 * np.exp(r * T), strikes[i], T,
                                            r, ATM_iv, x[0])
        err += (vol - impliedCallVolatility(S_0, strikes[i], T, r, price))**2

    return err


df = df_smile_summary.copy()
df.rename(columns={'market_iv': 'impliedvol'}, inplace=True)
initialGuess = [0.5]
res = least_squares(
    lambda x: error(
        x,
        df['strike'].values,
        df['impliedvol'].values,
    ), initialGuess)
beta = res.x[0]

displaced_diffusion_calibration = []
for K in df_smile_summary['strike']:
    calibration_vol = brentq(
        lambda x: op.Black_Scholes_call(S_0, K, T, r, x) - op.
        displaced_diffusion_call(S_0 * np.exp(r * T), K, T, r, ATM_iv, beta),
        1e-6, 1)
    displaced_diffusion_calibration.append([K, calibration_vol])
df_temp = pd.DataFrame(displaced_diffusion_calibration,
                       columns=['strike', 'calibration_vol'])
ax1.plot(df_temp['strike'],
         df_temp['calibration_vol'],
         '-',
         label='displaced diffusion model(beta=%s)' % np.round(beta, 4),
         linewidth=1)
#-------------------------------SABR calibration--------------------------------
F = F_0


def sabrcalibration(x, strikes, vols, F, T):
    err = 0.0
    for i, vol in enumerate(vols):
        err += (vol - SABR(F, strikes[i], T, x[0], 0.8, x[1], x[2]))**2

    return err


df = df_smile_summary.copy()
df.rename(columns={'market_iv': 'impliedvol'}, inplace=True)
initialGuess = [0.02, 0.2, 0.1]
res = least_squares(
    lambda x: sabrcalibration(x, df['strike'].values, df['impliedvol'].values,
                              F, T), initialGuess)
alpha = res.x[0]
beta = 0.8
rho = res.x[1]
nu = res.x[2]

SABR_calibration = []
for K in df_smile_summary['strike']:
    calibration_vol = SABR(F, K, T, alpha, beta, rho, nu)
    SABR_calibration.append([K, calibration_vol])
df_temp2 = pd.DataFrame(SABR_calibration,
                        columns=['strike', 'calibration_vol'])
ax1.plot(df_temp2['strike'],
         df_temp2['calibration_vol'],
         '-',
         label='SABR model',
         linewidth=1)
ax1.legend(loc=0)
figure1.savefig('part2_graph.png')
plt.show()