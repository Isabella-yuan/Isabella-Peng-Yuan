# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 21:05:04 2019

@author: AS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import option_pricing as op
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, least_squares
from scipy.interpolate import interp1d
from datetime import date

d1 = date(2013,8,30)
d2 = date(2015,1,17)
T = (d2-d1).days
#print(T)

ratedf=pd.read_csv('discount.csv')
x=ratedf['Day']
y=ratedf['Rate (%)']
fig1=plt.figure(figsize=(8,5))
plt.plot(x,y)
plt.xlabel('day')
plt.ylabel('interest rate')
plt.title('Term Structure')
plt.grid()
plt.savefig('TermStructure1.png')
plt.show()

p=np.polyfit(x,y,3)

#T=505
r1=p[0]*T**3+p[1]*T**2+p[2]*T+p[3]
r=round(r1/100,3)

def BlackscholesValue(r, T, S, sigma):
    return np.exp(-r*T)*((S**3)*np.exp((3*r+3*sigma**2)*T))*(10**(-8))+0.5*(np.log(S)+(r-0.5*sigma**2)*T)

def impliedPutVolatility(S_0, K, T, r, price):
    implied_Vol = brentq(
        lambda x: price - op.bs_euro_vanilla_put(S_0, K, T, r, x), 1e-6, 1)
    return implied_Vol

S_0=846.9
K=850
T=505/365
#r=0.004
goog_put=pd.read_csv('goog_put.csv', index_col=0, parse_dates=True)
for i in range(len(goog_put['strike'])):
    if goog_put['strike'][i]==850:
        price=(goog_put['best_bid'][i]+goog_put['best_offer'][i])/2
        break

impvol=impliedPutVolatility(S_0, K, T, r, price)

#black scholes
BS0=S_0**3*1e-8*np.exp(2*r*T+3*impvol**2*T)+0.5*np.exp(-r*T)*(np.log(S_0)+(r-0.5*impvol**2)*T)+10*np.exp(-r*T)
BS0

#bachelier
np.random.seed(0)
wt=np.random.normal(0,np.sqrt(505/365),10000)
st=S_0*(1+impvol*wt)

stsim=[]
for i in range(10000):
    x=st[i]**3*1e-8+0.5*np.log(st[i])+10
    stsim.append(x)

B01=np.nanmean(stsim)

print(BS0,B01)
#BS0: black scholes price, B0: bachelier price






