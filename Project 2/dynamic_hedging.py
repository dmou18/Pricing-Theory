from re import U
from tkinter import NS
from turtle import back
from unicodedata import name
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pdb

class BS():
    def SimStock(Nsims, Nsteps, S_0, T, mu, sigma):
        dt = T / Nsteps
        BM = np.full((Nsims, Nsteps), np.nan)
        sim_paths = np.full((Nsims, Nsteps), np.nan)
        
        BM[:,0] = 0
        sim_paths[:,0] = S_0
        BM[:, 1:] = np.cumsum(norm.rvs(size= (Nsims, Nsteps-1), scale=np.sqrt(dt)), axis=1)
        
        for i in range(Nsteps):
            sim_paths[:, i] = S_0*np.exp((mu - 0.5*np.square(sigma))*dt*i +sigma*BM[:, i])
        
        return sim_paths
    
    
    def CallPrice(t, S, T, K, sigma, r):
        
        dp = (np.log(S/K) + (r+0.5*sigma**2)*(T-t))/(np.sqrt((T-t))*sigma)
        dm = (np.log(S/K) + (r-0.5*sigma**2)*(T-t))/(np.sqrt((T-t))*sigma)
        
        return S*norm.cdf(dp) - K*np.exp(-r*(T-t))*norm.cdf(dm)
    
    
    def PutPrice(t, S, T, K, sigma, r):
        
        dp = (np.log(S/K) + (r+0.5*sigma**2)*(T-t))/np.sqrt((T-t))/sigma
        dm = (np.log(S/K) + (r-0.5*sigma**2)*(T-t))/np.sqrt((T-t))/sigma
        
        return K*np.exp(-r*(T-t))*norm.cdf(-dm) - S*norm.cdf(-dp)
    
    
    def CallDelta(t, S, T, K, sigma, r):
        
        dp = (np.log(S/K) + (r+0.5*sigma**2)*(T-t))/np.sqrt((T-t))/sigma
        
        return norm.cdf(dp)
    
    def PutDelta(t, S, T, K, sigma, r):
        
        return BS.CallDelta(t, S, T, K, sigma, r)-1
    
    def CallGamma(t, S, T, K, sigma, r):
        
        dp = (np.log(S/K) + (r+0.5*sigma**2)*(T-t))/np.sqrt((T-t))/sigma
        
        return norm.pdf(dp)/(S*sigma*np.sqrt((T-t)))
    
    def PutGamma(t, S, T, K, sigma, r):
        
        return BS.CallGamma(t, S, T, K, sigma, r)
    
    
class Dynamic_Hedging():
    def DeltaHedging(spotPrice, Nsteps, T, K, sigma, r, transCost):
        dt = T/Nsteps
        timeVector = np.linspace(0, T, Nsteps)
        optionPrice = BS.PutPrice(timeVector, spotPrice, T, K, sigma, r) 
        delta = BS.PutDelta(timeVector, spotPrice, T, K, sigma, r)
        bankAccount = np.full_like(spotPrice, np.nan)
        compounding = np.exp(r*dt)
        
        bankAccount[:, 0] = optionPrice[:, 0]-delta[:, 0]*spotPrice[:, 0]-np.abs(delta[:, 0])*transCost
        for i in range(1, Nsteps-1):
            bankAccount[:,i] = bankAccount[:, i-1]*compounding - (delta[:,i]-delta[:,i-1])*spotPrice[:,i] - np.abs(delta[:,i]-delta[:,i-1])*transCost
        
        payoff = K-spotPrice[:,-1]
        payoff[payoff < 0] = 0
        bankAccount[:,-1] = bankAccount[:,-2]*compounding - payoff + delta[:,-2]*spotPrice[:,-1] - np.abs(delta[:,-2])*transCost
        
        return delta, bankAccount
    
    
    def MoveBasedDeltaHedging(spotPrice, Nsteps, T, K, sigma, r, transCost, bandwidth = 0.05):
        dt = T/Nsteps
        timeVector = np.linspace(0, T, Nsteps)
        optionPrice = BS.PutPrice(timeVector, spotPrice, T, K, sigma, r)
        delta = BS.PutDelta(timeVector, spotPrice, T, K, sigma, r)
        bankAccount = np.full_like(spotPrice, np.nan)
        lazyDelta = np.full_like(delta, np.nan)
        compounding = np.exp(r*dt)
        
        lazyDelta[:,0] = delta[:,0]
        uband = delta[:,0]+bandwidth
        lband = delta[:,0]-bandwidth
        uband[uband > -0.01] = -0.01
        lband[lband < -0.99] = -0.99
        
        for i in range(1, Nsteps-1):
            lazyDelta[:,i] = lazyDelta[:,i-1]
            update_indx = np.any([delta[:,i] > uband, delta[:,i] < lband], axis=0)
            lazyDelta[:,i][update_indx] = delta[:,i][update_indx]
            
            uband[update_indx] = lazyDelta[:,i][update_indx]+bandwidth
            lband[update_indx] = lazyDelta[:,i][update_indx]-bandwidth
            uband[uband > -0.01] = -0.01
            lband[lband < -0.99] = -0.99
            
            temp = lazyDelta[:,i]
            temp[temp > -0.01] = -0.01
            temp[temp < -0.99] = -0.99
            lazyDelta[:,i] = temp
            
        bankAccount[:, 0] = optionPrice[:, 0]-delta[:, 0]*spotPrice[:, 0]-np.abs(delta[:, 0])*transCost
        for i in range(1, Nsteps-1):
            bankAccount[:,i] = bankAccount[:, i-1]*compounding - (lazyDelta[:,i]-lazyDelta[:,i-1])*spotPrice[:,i] - np.abs(lazyDelta[:,i]-lazyDelta[:,i-1])*transCost
        
        payoff = K-spotPrice[:,-1]
        payoff[payoff < 0] = 0
        bankAccount[:,-1] = bankAccount[:,-2] - payoff + lazyDelta[:,-2]*spotPrice[:,-1] - np.abs(lazyDelta[:,-2])*transCost
        
        return lazyDelta, bankAccount
        
    
    def DeltaGammaHedging(spotPrice, Nsteps, T1, T2, K, sigma, r, equityTransCost, optTransCost):
        dt = T1/Nsteps
        timeVector = np.linspace(0, T1, Nsteps)
        putOption = BS.PutPrice(timeVector, spotPrice, T1, K, sigma, r)
        callOption = BS.CallPrice(timeVector, spotPrice, T2, K, sigma, r)
        bankAccount = np.full_like(spotPrice, np.nan)
        compounding = np.exp(r*dt)
        putDelta = BS.PutDelta(timeVector, spotPrice, T1, K, sigma, r)
        callDelta = BS.CallDelta(timeVector, spotPrice, T2, K, sigma, r)
        putGamma = BS.PutGamma(timeVector, spotPrice, T1, K, sigma, r)
        callGamma = BS.CallGamma(timeVector, spotPrice, T2, K, sigma, r)
        
        gamma = putGamma/callGamma
        alpha = putDelta - gamma*callDelta
        
        bankAccount[:,0] = putOption[:, 0]-alpha[:, 0]*spotPrice[:, 0]-np.abs(alpha[:, 0])*equityTransCost-gamma[:,0]*callOption[:,0]\
            -np.abs(gamma[:,0])*optTransCost
        for i in range(1, Nsteps-1):
            bankAccount[:,i] = bankAccount[:, i-1]*compounding - (alpha[:,i]-alpha[:,i-1])*spotPrice[:,i] - np.abs(alpha[:,i]-alpha[:,i-1])*equityTransCost \
                - (gamma[:,i] - gamma[:,i-1])-np.abs(gamma[:,i] - gamma[:,i-1])*optTransCost
        
        putPayoff = K-spotPrice[:,-1]
        putPayoff[putPayoff < 0] = 0
        
        bankAccount[:,-1] = bankAccount[:,-2]*compounding - putPayoff + alpha[:,-2]*spotPrice[:,-1] - np.abs(alpha[:,-2])*equityTransCost \
            + gamma[:,-2]*callOption[:,-1] - np.abs(gamma[:,-2])*optTransCost
            
        return alpha, gamma, bankAccount
    
    
    def MoveBasedDeltaGammaHedging(spotPrice, Nsteps, T1, T2, K, sigma, r, equityTransCost, optTransCost, bandwidth):
        dt = T1/Nsteps
        timeVector = np.linspace(0, T1, Nsteps)
        putOption = BS.PutPrice(timeVector, spotPrice, T1, K, sigma, r)
        callOption = BS.CallPrice(timeVector, spotPrice, T2, K, sigma, r)
        bankAccount = np.full_like(spotPrice, np.nan)
        compounding = np.exp(r*dt)
        putDelta = BS.PutDelta(timeVector, spotPrice, T1, K, sigma, r)
        callDelta = BS.CallDelta(timeVector, spotPrice, T2, K, sigma, r)
        putGamma = BS.PutGamma(timeVector, spotPrice, T1, K, sigma, r)
        callGamma = BS.CallGamma(timeVector, spotPrice, T2, K, sigma, r)
        
        gamma = putGamma/callGamma
        alpha = putDelta - gamma*callDelta
        lazyAlpha = np.full_like(alpha, np.nan)
        lazyGamma = np.full_like(gamma, np.nan)
        
        lazyAlpha[:,0] = alpha[:,0]
        lazyAlpha[:,0] = gamma[:,0]
        uband = putDelta[:,0]+bandwidth
        lband = putDelta[:,0]-bandwidth
        uband[uband > -0.01] = -0.01
        lband[lband < -0.99] = -0.99
        
        for i in range(1, Nsteps-1):
            lazyAlpha[:,i] = lazyAlpha[:,i-1]
            lazyGamma[:,i] = lazyGamma[:,i-1]
            update_indx = np.any([putDelta[:,i] > uband, putDelta[:,i] < lband], axis=0)
            lazyAlpha[:,i][update_indx] = alpha[:,i][update_indx]
            lazyGamma[:,i][update_indx] = gamma[:,i][update_indx]
            
            
            uband[update_indx] = putDelta[:,i][update_indx]+bandwidth
            lband[update_indx] = putDelta[:,i][update_indx]-bandwidth
            uband[uband > -0.01] = -0.01
            lband[lband < -0.99] = -0.99
            
            temp = lazyAlpha[:,i]
            temp[temp > -0.01] = -0.01
            temp[temp < -0.99] = -0.99
            lazyAlpha[:,i] = temp
            
            temp = lazyGamma[:,i]
            temp[temp > -0.01] = -0.01
            temp[temp < -0.99] = -0.99
            lazyGamma[:,i] = temp
            
        bankAccount[:,0] = putOption[:, 0]-lazyAlpha[:, 0]*spotPrice[:, 0]-np.abs(lazyAlpha[:, 0])*equityTransCost-lazyGamma[:,0]*callOption[:,0]\
            -np.abs(lazyGamma[:,0])*optTransCost
        for i in range(1, Nsteps-1):
            bankAccount[:,i] = bankAccount[:, i-1]*compounding - (lazyAlpha[:,i]-lazyAlpha[:,i-1])*spotPrice[:,i] - np.abs(lazyAlpha[:,i]-lazyAlpha[:,i-1])*equityTransCost \
                - (lazyGamma[:,i] - lazyGamma[:,i-1])-np.abs(lazyGamma[:,i] - lazyGamma[:,i-1])*optTransCost
        
        putPayoff = K-spotPrice[:,-1]
        putPayoff[putPayoff < 0] = 0
        
        bankAccount[:,-1] = bankAccount[:,-2]*compounding - putPayoff + lazyAlpha[:,-2]*spotPrice[:,-1] - np.abs(lazyAlpha[:,-2])*equityTransCost \
            + lazyGamma[:,-2]*callOption[:,-1] - np.abs(lazyGamma[:,-2])*optTransCost
            
        return lazyAlpha, lazyGamma, bankAccount
            
            
        
        
    
    
class Analysis():
    def PlotDeltaHedging(spotPrice, Nsteps, T, K, sigma, r, equityTransCost, bandwidth):
        timeVector = np.linspace(0, T, Nsteps)
        delta, bankAccount = Dynamic_Hedging.DeltaHedging(spotPrice, Nsteps, T, K, sigma, r, equityTransCost)
        lazyDelta, lazyBankAccount = Dynamic_Hedging.MoveBasedDeltaHedging(spotPrice, Nsteps, T, K, sigma, r, equityTransCost, bandwidth)
        
        #Plot simulation paths for stock process
        fig = plt.figure(1)
        fig.suptitle("Spot Price Paths")
        plt.xlabel('Time')
        plt.ylabel('Spot Price')
        plt.plot(timeVector, spotPrice.T)
        
        #Plot simulation paths for delta
        fig = plt.figure(2)
        fig.suptitle("Time-based Delta Position for Delta Hedging")
        plt.xlabel('Time')
        plt.ylabel(r'$\Delta$')
        plt.plot(timeVector, delta.T)
        
        #Plot simulation paths for bank account
        fig = plt.figure(3)
        fig.suptitle("Time-based Bank Account for Delta Hedging")
        plt.xlabel('Time')
        plt.ylabel('Bank Account Value')
        plt.plot(timeVector, bankAccount.T)
        
        # #Plot simulation paths for move-based delta
        fig = plt.figure(4)
        fig.suptitle("Move-based Delta Position for Delta Hedging")
        plt.xlabel('Time')
        plt.ylabel(r'$\Delta$')
        plt.plot(timeVector, lazyDelta.T)
        
        #Plot simulation paths for move-based bank account
        fig = plt.figure(5)
        fig.suptitle("Move-based Bank Account for Delta Hedging")
        plt.xlabel('Time')
        plt.ylabel('Bank Account Value')
        plt.plot(timeVector, lazyBankAccount.T)
        
        plt.show()
        
        
    def PlotDeltaGammaHedging(spotPrice, Nsteps, T1, T2, K, sigma, r, equityTransCost, optTransCost, bandwidth):
        timeVector = np.linspace(0, T1, Nsteps)
        
        alpha, gamma, bankAccount = Dynamic_Hedging.DeltaGammaHedging(spotPrice, Nsteps, T1, T2, K, sigma, r, equityTransCost, optTransCost)
        lazyAlpha, lazyGamma, lazyBankAccount = Dynamic_Hedging.MoveBasedDeltaGammaHedging(spotPrice, Nsteps, T1, T2, K, sigma, r, equityTransCost, optTransCost, bandwidth)
        
        #Plot simulation paths for stock process
        fig = plt.figure(1)
        fig.suptitle("Spot Price Paths")
        plt.xlabel('Time')
        plt.ylabel('Spot Price')
        plt.plot(timeVector, spotPrice.T)
        
        #Plot simulation paths for delta
        fig = plt.figure(2)
        fig.suptitle("Time-based Delta Position for Delta-Gamma Hedging")
        plt.xlabel('Time')
        plt.ylabel(r'$\Delta$')
        plt.plot(timeVector, alpha.T)
        
        #Plot simulation paths for gamma
        fig = plt.figure(3)
        fig.suptitle("Time-based Gamma Position for Delta-Gamma Hedging")
        plt.xlabel('Time')
        plt.ylabel(r'$\gamma$')
        plt.plot(timeVector, gamma.T)
        
        #Plot simulation paths for bank account
        fig = plt.figure(4)
        fig.suptitle("Time-based Bank Account for Delta-Gamma Hedging")
        plt.xlabel('Time')
        plt.ylabel('Bank Account Value')
        plt.plot(timeVector, bankAccount.T)
        
        # #Plot simulation paths for delta
        # fig = plt.figure(5)
        # fig.suptitle("Move-based Delta Position for Delta-Gamma Hedging")
        # plt.xlabel('Time')
        # plt.ylabel(r'$\Delta$')
        # plt.plot(timeVector, lazyAlpha.T)
        
        # #Plot simulation paths for gamma
        # fig = plt.figure(6)
        # fig.suptitle("Move-based Gamma Position for Delta-Gamma Hedging")
        # plt.xlabel('Time')
        # plt.ylabel(r'$\gamma$')
        # plt.plot(timeVector, lazyGamma.T)
        
        # #Plot simulation paths for bank account
        # fig = plt.figure(7)
        # fig.suptitle("Move-based Bank Account for Delta-Gamma Hedging")
        # plt.xlabel('Time')
        # plt.ylabel('Bank Account Value')
        # plt.plot(timeVector, lazyBankAccount.T)
        
        plt.show()
    
        


if __name__ == "__main__":
    equityTransCost= 0.005
    optTransCost = 0.01
    T1 = 0.25
    T2 = 0.5
    Nsteps = 1000
    Nsims = 1
    mu = 0.1
    sigma = 0.2
    r = 0.02
    K = 100
    S_0 = 100
    bandwidth = 0.05
    
    
    spotPrice = BS.SimStock(Nsims, Nsteps, S_0, T1, mu, sigma)
    
    Analysis.PlotDeltaHedging(spotPrice, Nsteps, T1, K, sigma, r, equityTransCost, bandwidth)    
    #Analysis.PlotDeltaGammaHedging(spotPrice, Nsteps, T1, T2, K, sigma, r, equityTransCost, optTransCost, bandwidth)
