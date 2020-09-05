#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:41:17 2019

@author: huyuemei
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pylab as plt
from matplotlib.ticker import PercentFormatter

def DynamicHedge(stepN,paths=50000,S0=100,sigma=0.2,r=0.05,T=1/12,K=100):
    def simulate_Brownian_Motion(paths, steps, T):
        deltaT = T/steps
        t = np.linspace(0, T, steps+1)
        X = np.c_[np.zeros((paths, 1)),
                  np.random.randn(paths, steps)]
        return t, np.cumsum(np.sqrt(deltaT) * X, axis=1)
    def BlackScholesCall(S, K, r, sigma, T):
        d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

#    S0=100
#    sigma=0.2
#    r=0.05
#    T=1/12
#    K=100
    
    #simulate Brownian motion
    t, x = simulate_Brownian_Motion(paths, stepN, T)
    
    #generate stock price
    St=S0*np.exp((r-0.5*sigma**2)*t+sigma*x)
    error=0
    
    #calculate option premium received in t=0
    c0=BlackScholesCall(S0,K,r,sigma,T)
    
    
    #calculate option payoff at t=T
    option_payoff=np.maximum(St[:,-1]-K,0)
    
    #calculate dynamic hedging total P&L 
    for i in range(0,stepN):
        d1 = (np.log(St[:,i]/K)+(r+sigma**2/2)*(T-t[i])) / (sigma*np.sqrt(T-t[i])) #from standing point to strike point
        d2 = d1 - sigma*np.sqrt(T-t[i]) #for i, num of d1:50000
        Phi_all=norm.cdf(d1) #50000
        PsiBt_all= -K*np.exp(-r*(T-t[i]))*norm.cdf(d2)
        port_v_t=Phi_all*St[:,i]+PsiBt_all
        port_v_T=Phi_all*St[:,i+1]+PsiBt_all*np.exp(r*(t[i+1]-t[i]))
        
        error+=port_v_T-port_v_t
    
    #total error
    total_error=error+c0-option_payoff
    
    return total_error,c0

hedge21 = DynamicHedge(21)[0]
hedge84 = DynamicHedge(84)[0]

#plot
fig1=plt.figure(figsize=(12,5))
fig1.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False,
                left=False, right=False)
plt.xlabel('measured volatility')

ax1=fig1.add_subplot(121)
ax1.hist(hedge21,weights=np.ones(len(hedge21))/len(hedge21),
         bins=np.linspace(-2,2,35))
ax1.yaxis.set_major_formatter(PercentFormatter(1))
#ax1.set_xlabel('measured volatility')
ax1.set_ylabel('Frequency (out of 100%)')
ax1.set_title('Dynamic Hedge error for N=21')
ax1.yaxis.grid(color='gray', linestyle='dashed')

ax2=fig1.add_subplot(122,sharey=ax1)
ax2.hist(hedge84,weights=np.ones(len(hedge84))/len(hedge84),
         bins=np.linspace(-2,2,35))
ax2.set_title('Dynamic Hedge error for N=84')
ax2.yaxis.grid(color='gray', linestyle='dashed')
plt.tight_layout()

plt.savefig('hedgeerror.png')
plt.show()

#statistics
mean=[]
std=[]
std_ratio=[]
c0=DynamicHedge(21)[1]
for i in [hedge21,hedge84]:
    mean.append(np.mean(i))
    std.append(np.std(i,ddof=1))
    std_ratio.append(np.std(i,ddof=1)/c0)

data=pd.DataFrame(index=['N=21','N=84'])
data['Mean P&L']=np.array(mean).round(3)
data['Standard Dev. of P&L']=np.array(std).round(2)
data['StDev of P&L as a % of option premium']=np.array(std_ratio).round(4)*100
print(data)