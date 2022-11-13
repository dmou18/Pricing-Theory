import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pdb
from black_scholes import BS

class Dynamic_Hedging():
    def DeltaHedging(spotPrice, Nsteps, T, dt, K, sigma, r, transCost, settle=False):
        timeVector = T - np.linspace(0, T, Nsteps+1)
        optionPrice = BS.PutPrice(spotPrice, timeVector, K, sigma, r) 
        delta = BS.PutDelta(spotPrice, timeVector, K, sigma, r)
        numTrades = np.full(spotPrice.shape[0], 0, dtype='float64')
        bankAccount = np.full_like(spotPrice, np.nan)
        compounding = np.exp(r*dt)
        
        bankAccount[:,0] = optionPrice[:,0] - delta[:, 0]*spotPrice[:,0] - np.abs(delta[:,0])*transCost
        
        for i in range(1, Nsteps):
            bankAccount[:,i] = bankAccount[:,i-1]*compounding - (delta[:,i]-delta[:,i-1])*spotPrice[:,i] - np.abs(delta[:,i]-delta[:,i-1])*transCost
            numTrades += np.abs(delta[:,i]-delta[:,i-1]) 
        
        if settle:
            payoff = np.maximum(K-spotPrice[:,-1],0)
            bankAccount[:,-1] = bankAccount[:,-2]*compounding - payoff + delta[:,-2]*spotPrice[:,-1] - np.abs(delta[:,-2])*transCost
            numTrades += np.abs(delta[:,-2])
            delta[:,-1] = 0
        else:
            bankAccount[:,-1] = bankAccount[:,-2]*compounding
            delta[:,-1] = delta[:,-2]
        return delta, bankAccount, optionPrice, numTrades
    
    
    def MoveBasedDeltaHedging(spotPrice, Nsteps, T, dt, K, sigma, r, transCost, bandwidth = 0.05, settle=False):
        timeVector = T - np.linspace(0, T, Nsteps+1)
        optionPrice = BS.PutPrice(spotPrice, timeVector, K, sigma, r)
        delta = BS.PutDelta(spotPrice, timeVector, K, sigma, r)
        bankAccount = np.full_like(spotPrice, np.nan)
        numTrades = np.full(spotPrice.shape[0], 0, dtype='float64')
        lazyDelta = np.full_like(delta, np.nan)
        compounding = np.exp(r*dt)
        
        lazyDelta[:,0] = delta[:,0]
        
        uband = delta[:,0] + bandwidth
        lband = uband - 2*bandwidth
        
        for i in range(1, Nsteps+1):
            lazyDelta[:,i] = lazyDelta[:,i-1]
            
            update_index = np.any([delta[:,i] > uband, delta[:,i] < lband], axis=0)
            lazyDelta[:,i][update_index] = delta[:,i][update_index]
            
            uband[update_index] = delta[:,i][update_index] + bandwidth
            lband = uband - 2*bandwidth
            
            u = uband > -0.01
            d = lband < -0.99
            uband[u] = -0.01
            lband[d] = -0.99
            
            u2 = delta[:,i] > -0.01
            d2 = delta[:,i] < -0.99
            uband[u2] = 0
            lband[d2] = -1
            
            numTrades += np.abs(lazyDelta[:,i]-lazyDelta[:,i-1])
            
        bankAccount[:, 0] = optionPrice[:,0]-delta[:,0]*spotPrice[:,0]-np.abs(delta[:,0])*transCost
        for i in range(1, Nsteps):
            bankAccount[:,i] = bankAccount[:, i-1]*compounding - (lazyDelta[:,i]-lazyDelta[:,i-1])*spotPrice[:,i] - np.abs(lazyDelta[:,i]-lazyDelta[:,i-1])*transCost
        
        if settle:
            payoff = np.maximum(K-spotPrice[:,-1],0)
            bankAccount[:,-1] = bankAccount[:,-2]*compounding - payoff + lazyDelta[:,-2]*spotPrice[:,-1] - np.abs(lazyDelta[:,-2])*transCost
            lazyDelta[:,-1] = 0
            numTrades += np.abs(delta[:,-2])
        else:
            bankAccount[:,-1] = bankAccount[:,-2]*compounding
            lazyDelta[:,-1] = lazyDelta[:,-2]
        
        return lazyDelta, bankAccount, optionPrice, numTrades      
    
    
    def DeltaGammaHedging(spotPrice, Nsteps, T1, T2, dt, K, sigma, r, equityTransCost, optTransCost, settle = False):
        timeVector = np.linspace(0, T1, Nsteps + 1)
        putOption = BS.PutPrice(spotPrice, T1 -  timeVector, K, sigma, r)
        callOption = BS.CallPrice(spotPrice, T2 - timeVector, K, sigma, r)
        putDelta = BS.PutDelta(spotPrice, T1 - timeVector, K, sigma, r)
        callDelta = BS.CallDelta(spotPrice, T2 - timeVector, K, sigma, r)
        putGamma = BS.PutGamma(spotPrice, T1 - timeVector, K, sigma, r)
        callGamma = BS.CallGamma(spotPrice, T2 - timeVector, K, sigma, r)
        
        bankAccount = np.full_like(spotPrice, np.nan)
        compounding = np.exp(r*dt)
        gamma = putGamma/callGamma
        alpha = putDelta - gamma*callDelta
        
        bankAccount[:,0] = putOption[:, 0] - alpha[:, 0]*spotPrice[:, 0] -  np.abs(alpha[:, 0])*equityTransCost\
            - gamma[:,0]*callOption[:,0] - np.abs(gamma[:,0])*optTransCost
        for i in range(1, Nsteps):
            bankAccount[:,i] = bankAccount[:, i-1]*compounding - (alpha[:,i]-alpha[:,i-1])*spotPrice[:,i] \
                - np.abs(alpha[:,i]-alpha[:,i-1])*equityTransCost \
                - (gamma[:,i]-gamma[:,i-1])*callOption[:,i] - np.abs(gamma[:,i] - gamma[:,i-1])*optTransCost
        
        if settle:
            putPayoff = np.maximum(K-spotPrice[:,-1],0)
            bankAccount[:,-1] = bankAccount[:,-2]*compounding - putPayoff + alpha[:,-2]*spotPrice[:,-1] \
                - np.abs(alpha[:,-2])*equityTransCost + gamma[:,-2]*callOption[:,-1] - np.abs(gamma[:,-2])*optTransCost
            alpha[:,-1] = 0
            gamma[:,-1] = 0
        else:
            bankAccount[:,-1] = bankAccount[:,-2]*compounding
            alpha[:,-1] = alpha[:,-2]
            gamma[:,-1] = gamma[:,-2]
            
        return alpha, gamma, callOption, putOption, bankAccount
    
    
    def MoveBasedDeltaGammaHedging(spotPrice, Nsteps, T1, T2, dt, K, sigma, r, equityTransCost, optTransCost, bandwidth, settle=False):
        dt = T1/Nsteps
        timeVector = np.linspace(0, T1, Nsteps + 1)
        putOption = BS.PutPrice(spotPrice, T1 -  timeVector, K, sigma, r)
        callOption = BS.CallPrice(spotPrice, T2 - timeVector, K, sigma, r)
        putDelta = BS.PutDelta(spotPrice, T1 - timeVector, K, sigma, r)
        callDelta = BS.CallDelta(spotPrice, T2 - timeVector, K, sigma, r)
        putGamma = BS.PutGamma(spotPrice, T1 - timeVector, K, sigma, r)
        callGamma = BS.CallGamma(spotPrice, T2 - timeVector, K, sigma, r)
        
        bankAccount = np.full_like(spotPrice, np.nan)
        compounding = np.exp(r*dt)
        
        gamma =  putGamma/callGamma
        alpha = putDelta - gamma*callDelta
        lazyAlpha = np.full_like(alpha, np.nan)
        lazyGamma = np.full_like(gamma, np.nan)
        
        lazyAlpha[:,0] = alpha[:,0]
        lazyGamma[:,0] = gamma[:,0]
        
        uband = putDelta[:,0]+bandwidth
        lband = uband-2*bandwidth
        
        for i in range(1, Nsteps+1):
            lazyAlpha[:,i] = lazyAlpha[:,i-1]
            lazyGamma[:,i] = lazyGamma[:,i-1]
            
            update_index = np.any([putDelta[:,i] > uband, putDelta[:,i] < lband], axis=0)
            
            lazyAlpha[:,i][update_index] = alpha[:,i][update_index]
            lazyGamma[:,i][update_index] = gamma[:,i][update_index]
            
            uband[update_index] = putDelta[:,i][update_index] + bandwidth
            lband = uband - 2*bandwidth
            
            u = uband > -0.01
            d = lband < -0.99
            uband[u] = -0.01
            lband[d] = -0.99
            
            u2 = putDelta[:,i] > -0.01
            d2 = putDelta[:,i] < -0.99
            uband[u2] = 0
            lband[d2] = -1
    
        bankAccount[:,0] = putOption[:, 0]-lazyAlpha[:, 0]*spotPrice[:, 0]-np.abs(lazyAlpha[:, 0])*equityTransCost\
            -lazyGamma[:,0]*callOption[:,0] - np.abs(lazyGamma[:,0])*optTransCost

        for i in range(1, Nsteps):
            bankAccount[:,i] = bankAccount[:, i-1]*compounding - (lazyAlpha[:,i]-lazyAlpha[:,i-1])*spotPrice[:,i]\
                - np.abs(lazyAlpha[:,i]-lazyAlpha[:,i-1])*equityTransCost - (lazyGamma[:,i]-lazyGamma[:,i-1])*callOption[:,i]\
                - np.abs(lazyGamma[:,i] - lazyGamma[:,i-1])*optTransCost
                
        if settle:
            putPayoff = np.maximum(K-spotPrice[:,-1],0)
            bankAccount[:,-1] = bankAccount[:,-2]*compounding - putPayoff + lazyAlpha[:,-2]*spotPrice[:,-1] \
                + lazyGamma[:,-2]*callOption[:,-1] - np.abs(lazyAlpha[:,-2])*equityTransCost - np.abs(lazyGamma[:,-2])*optTransCost
            lazyAlpha[:,-1] = 0
            lazyGamma[:,-1] = 0
        else:
            bankAccount[:,-1] = bankAccount[:,-2]*compounding 
            lazyAlpha[:,-1] = lazyAlpha[:,-2]
            lazyGamma[:,-1] = lazyGamma[:,-2]
            
        return lazyAlpha, lazyGamma, callOption, putOption, bankAccount 