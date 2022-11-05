# %%
import numpy as np
import csv
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import pdb

from black_scholes import BS
from dynamic_hedging import Dynamic_Hedging
from analysis import Analysis


if __name__ == "__main__": 
    equityTransCost= 0.005*1
    optTransCost = 0.01*1
    T1 = 0.25
    T2 = 0.5
    dt = 1/356
    Nsteps = int(T1/dt)
    Nsims = 10000
    mu = 0.1
    sigma = 0.2
    r = 0.02
    K = 100
    S_0 = 100
    bandwidth = 0.05
    c_level = 0.1
    benchmark = -0.02
    
    new_sim = False
    settle = True
    
    spotPrice = None
    if new_sim:
        spotPrice = BS.SimStock(Nsims, Nsteps, dt, S_0, mu, sigma)
        with open("stock_sim.csv","w+") as my_csv:
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(spotPrice)
    else:
        df = pd.read_csv('stock_sim.csv', header=None)
        spotPrice = df.to_numpy()
        
    
    ''' Anaylysis for Delta Hedging '''
    # delta, bankAccount, option, numTrades = Dynamic_Hedging.DeltaHedging(spotPrice, Nsteps, T1, dt, K, sigma, r, equityTransCost, settle)
    # lazyDelta, lazyBankAccount, lazyOption, lazyNumTrades = Dynamic_Hedging.MoveBasedDeltaHedging(spotPrice, Nsteps, T1, dt, K, sigma, r, equityTransCost, bandwidth, settle)
    
    #Analysis.PlotDeltaHedging(spotPrice, Nsteps, T1, delta, bankAccount, lazyDelta, lazyBankAccount)  
    
    # portfolio = Analysis.deltaPort(spotPrice, delta, bankAccount, settle)
    # Analysis.plotPort(K, spotPrice[:,-1], portfolio, 'Portfolio Value before Settlement Time-based Delta Hedging', is_call = False)
    # Analysis.HistPL(portfolio, "P&L Distribution for Time-based Delta Hedging")
    # Analysis.HistTrades(numTrades, "Number of Unit of Underlying Asset Traded for Time-based Delta Hedging")
    
    # lazyPortfolio = Analysis.deltaPort(spotPrice, lazyDelta, lazyBankAccount, settle)
    # Analysis.plotPort(K, spotPrice[:,-1], lazyPortfolio, 'Portfolio Value before Settlement for Move-based Delta Hedging', is_call = False)
    # Analysis.HistPL(lazyPortfolio, "P&L Distribution for Move-based Delta Hedging")
    # Analysis.HistTrades(lazyNumTrades, "Number of Unit of Underlying Asset Traded for Move-based Delta Hedging")
    
    ''' Anaylysis for Delta-Gamma Hedging '''
    alpha, gamma, callOption, putOption, bankAccount = Dynamic_Hedging.DeltaGammaHedging(spotPrice, Nsteps, T1, T2, dt, K, sigma, r, equityTransCost, optTransCost, settle)
    lazyAlpha, lazyGamma, lazyCallOption, lazyPutOption, lazyBankAccount = Dynamic_Hedging.MoveBasedDeltaGammaHedging(spotPrice, Nsteps, T1, T2, dt, K, sigma, r, equityTransCost, optTransCost, bandwidth, settle)
    
    # Analysis.PlotDeltaGammaHedging(spotPrice, Nsteps, T1, alpha, gamma, bankAccount, lazyAlpha, lazyGamma, lazyBankAccount)
    
    portfolio = Analysis.deltaGammaPort(spotPrice, alpha, gamma, callOption, bankAccount, settle)
    # Analysis.plotPort(K, spotPrice[:,-1], portfolio, 'Portfolio Value before Settlement for Time-based Delta-Gamma Hedging', is_call = False)
    # Analysis.HistPL(portfolio, 'P&L Distribution for Time-based Delta-Gamma Hedging')
    
    # lazyPortfolio = Analysis.deltaGammaPort(spotPrice, lazyAlpha, lazyGamma, lazyCallOption, lazyBankAccount, settle)
    # Analysis.plotPort(K, spotPrice[:,-1], lazyPortfolio, 'Portfolio Value before Settlement for Move-based Delta-Gamma Hedging', is_call = False)
    # Analysis.HistPL(lazyPortfolio, 'P&L Distribution for Move-based Delta-Gamma Hedging')
    
    '''Calculate CVar'''
    CVaR, adjusted_price = Analysis.CVaR(portfolio, putOption[0,0], c_level, benchmark, r, T1)
    print(f"The CVaR for the portfolio at VaR level {c_level} is no larger than {benchmark} is {CVaR}")
    print(f"The Adjusted put option price so CVaR at VaR level {c_level} is no larger than {benchmark} is {adjusted_price}")

 # %%
