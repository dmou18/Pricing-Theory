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

np.seterr(divide='ignore', invalid='ignore')


if __name__ == "__main__": 
    equityTransCost= 0.005*1
    optTransCost = 0.01*1
    T1 = 0.25
    T2 = 0.5
    dt = 1/360
    Nsteps = int(T1/dt)
    Nsims = 10
    mu = 0.1
    sigma = 0.2
    r = 0.02
    K = 100
    S_0 = 100
    bandwidth = 0.05
    bandwidth_list = np.linspace(0, 0.2, 100)
    c_level = 0.1
    benchmark = -0.02
    
    settle = False
    
    # Start a new simulation if new_sim is true
    np.random.seed(89)
    spotPrice = BS.SimStock(Nsims, Nsteps, dt, S_0, mu, sigma)
    
    
    ''' Anaylysis for Delta Hedging '''
    # delta, bankAccount, option, numTrades = Dynamic_Hedging.DeltaHedging(spotPrice, Nsteps, T1, dt, K, sigma, r, equityTransCost, settle)
    # portfolio = Analysis.deltaPort(spotPrice, delta, bankAccount, settle)
    
    # lazyDelta, lazyBankAccount, lazyOption, lazyNumTrades = Dynamic_Hedging.MoveBasedDeltaHedging(spotPrice, Nsteps, T1, dt, K, sigma, r, equityTransCost, bandwidth, settle)
    # lazyPortfolio = Analysis.deltaPort(spotPrice, lazyDelta, lazyBankAccount, settle)
     
    # Analysis.PlotDeltaHedging(spotPrice, Nsteps, T1, delta, bankAccount, lazyDelta, lazyBankAccount)  
   
    # Analysis.plotPort(K, spotPrice[:,-1], portfolio, 'Portfolio Value before Settlement Time-based for Delta Hedging', is_call = False)
    # Analysis.HistPnL(portfolio, "P&L Distribution for Time-based Delta Hedging")
    # Analysis.HistTrades(numTrades, "Number of Unit of Underlying Asset Traded for Time-based Delta Hedging")
    
    # Analysis.plotPort(K, spotPrice[:,-1], lazyPortfolio, 'Portfolio Value before Settlement for Move-based Delta Hedging', is_call = False)
    # Analysis.HistPnL(lazyPortfolio, "P&L Distribution for Move-based Delta Hedging")
    # Analysis.HistTrades(lazyNumTrades, "Number of Unit of Underlying Asset Traded for Move-based Delta Hedging")
    
    # Analysis.HistPnL_2(portfolio1=portfolio, portfolio2=lazyPortfolio, label1="Time-based", label2="Move-based", title='P&L Distribution for Delta Hedging')
    
    
    ''' Anaylysis for Delta-Gamma Hedging '''
    alpha, gamma, callOption, putOption, bankAccount = Dynamic_Hedging.DeltaGammaHedging(spotPrice, Nsteps, T1, T2, dt, K, sigma, r, equityTransCost, optTransCost, settle)
    portfolio = Analysis.deltaGammaPort(spotPrice, alpha, gamma, callOption, bankAccount, settle)
    
    lazyAlpha, lazyGamma, lazyCallOption, lazyPutOption, lazyBankAccount = Dynamic_Hedging.MoveBasedDeltaGammaHedging(spotPrice, Nsteps, T1, T2, dt, K, sigma, r, equityTransCost, optTransCost, bandwidth, settle)
    lazyPortfolio = Analysis.deltaGammaPort(spotPrice, lazyAlpha, lazyGamma, lazyCallOption, lazyBankAccount, settle)
    
    Analysis.PlotDeltaGammaHedging(spotPrice, Nsteps, T1, alpha, gamma, bankAccount, lazyAlpha, lazyGamma, lazyBankAccount)
    
    # Analysis.plotPort(K, spotPrice[:,-1], portfolio, 'Portfolio Value before Settlement for Time-based Delta-Gamma Hedging', is_call = False)
    # Analysis.HistPnL(portfolio, 'P&L Distribution for Time-based Delta-Gamma Hedging')
    
    # Analysis.plotPort(K, spotPrice[:,-1], lazyPortfolio, 'Portfolio Value before Settlement for Move-based Delta-Gamma Hedging', is_call = False)
    # Analysis.HistPnL(lazyPortfolio, 'P&L Distribution for Move-based Delta-Gamma Hedging')
    
    # Analysis.HistPnL_2(portfolio1=portfolio, portfolio2=lazyPortfolio, label1="Time-based", label2="Move-based", title='P&L Distribution for Delta-Gamma Hedging')
    
    
    '''Calculate CVar'''
    # plot = True
    # portfolio = portfolio
    # optionPrice = option[0,0]
    
    # print("\nCVaR-adjusted option price for Time-based hedging portfolio:")
    # Analysis.CVaR(portfolio, optionPrice, c_level, benchmark, r, T1, plot)
    
    # print("\nCVaR-adjusted option price for Move-based hedging portfolio:")
    # lazyPortfolio = lazyPortfolio
    # lazyOptionPrice = option[0,0]
    # Analysis.CVaR(lazyPortfolio, lazyOptionPrice, c_level, benchmark, r, T1, plot)
    
    
    '''Efficeient Frontier for Different Bandwidth'''
    # Analysis.EfficientFrontier(spotPrice, Nsteps, T1, dt, K, sigma, r, equityTransCost, bandwidth_list, settle)
    # Analysis.EfficientFrontier_DG(spotPrice, Nsteps, T1, T2, dt, K, sigma, r, equityTransCost, optTransCost, bandwidth_list, settle)

 # %%
