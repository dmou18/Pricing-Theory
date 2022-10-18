# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 21:47:06 2022

@author: zhaiz, Dixin Mou
"""
from cmath import nan
from os import times
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Code for Q3 a
def PutOptionPricer(currStockPrice, strikePrice, intRate, vol, totSteps, yearsToExp):
    timeStep = yearsToExp / totSteps
    u = np.exp(intRate * timeStep + vol * np.sqrt(timeStep))
    # one step random walk (price decreases)
    d = np.exp(intRate * timeStep - vol * np.sqrt(timeStep))
    
    # risk neutral probability of an up move
    pu = 1/2 * (1 - 1/2 * vol * np.sqrt(timeStep))
    # risk neutral probability of a down move
    pd = 1 - pu
    
    priceTree = np.full((totSteps+1, totSteps+1), np.nan)
    euroOptionTree = np.full_like(priceTree, np.nan)
    americanOptionTree = np.full_like(priceTree, np.nan)
    boundary = np.full((totSteps+1, 1), np.nan)
    priceTree[0, 0] = currStockPrice
    
    for ii in range(1, totSteps+1):
        priceTree[0:ii, ii] = priceTree[0:ii, (ii-1)] * u
        priceTree[ii, ii] = priceTree[(ii-1), (ii-1)] * d

    euroOptionTree[:, -1] = np.maximum(0, strikePrice - priceTree[:, -1])
    americanOptionTree[:, -1] = np.maximum(0, strikePrice - priceTree[:, -1])
    boundary[-1] = priceTree[np.min(np.argwhere(euroOptionTree[:, -1] != 0)),-1]

    oneStepDiscount = np.exp(-intRate * timeStep) 
    backSteps = priceTree.shape[1] - 1
    
    for ii in range(backSteps, 0, -1):
        B = np.exp(intRate * timeStep * ii)
        euroOptionTree[0:ii, ii-1] =  B*oneStepDiscount *(pu * euroOptionTree[0:ii, ii]/B + pd * euroOptionTree[1:(ii+1), ii]/B)
        americanOptionTree[0:ii, ii-1] =  B*oneStepDiscount *(pu * americanOptionTree[0:ii, ii]/B + pd * americanOptionTree[1:(ii+1), ii]/B)
        
        inds = np.argwhere(strikePrice - priceTree[0:ii, ii-1] > americanOptionTree[0:ii, ii-1])
        if inds.size != 0:
            boundary[ii-1] = priceTree[np.min(inds),ii-1] 
            
        americanOptionTree[0:ii, ii-1] = np.maximum(strikePrice - priceTree[0:ii, ii-1], americanOptionTree[0:ii, ii-1])
       
    return americanOptionTree, euroOptionTree, priceTree, boundary


# Q3 Part(a) Generate the exercise boundary as a function of t
def pltBoundary(boundary, totSteps, yearsToExp):
    timeStep = yearsToExp / totSteps
    TimeVector = np.arange(0, yearsToExp + timeStep, timeStep)

    fig = plt.figure()
    fig.suptitle("Exercise Boundary for American Put Option")
    plt.plot(TimeVector, boundary)
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.show()

   
def hedgePortfolio(optionTree, priceTree, intRate, vol, totSteps, yearsToExp, american = False, plot = False):
    timeStep = yearsToExp/totSteps
    u = np.exp(intRate * timeStep + vol * np.sqrt(timeStep))
    d = np.exp(intRate * timeStep - vol * np.sqrt(timeStep))
    alphas = np.full((totSteps+1, 5), np.nan)
    betas = np.full((totSteps+1, 5), np.nan)
    stocks = np.full((totSteps+1, 5),np.nan)
    
    for i in range(5):
        ind = int(totSteps * i * 0.25)
        alpha = None
        beta = None
        S = priceTree[0:ind+1, ind]
        if i == 4:
            alpha = np.where(optionTree[:, ind] > 0, -1, 0)
            beta = (optionTree[:, ind] - (S*alpha))/np.exp(intRate * timeStep * ind)
        else:
            Cu = optionTree[0:ind+1, ind+1]
            Cd = optionTree[1:ind+2, ind+1]
            alpha = (Cu-Cd)/(S*(u-d))
            beta = (Cu - alpha*S*u)/np.exp(intRate * timeStep * ind)
        
        alphas[0:ind + 1,i] = alpha
        betas[0:ind + 1,i] = beta
        stocks[0 : ind + 1, i] = S
    
    if plot:
        f = plt.figure(1)
        plt.scatter(stocks[:, 0], alphas[:, 0], alpha=1, marker='o', linewidth=2, label=r'$t = 0$')
        plt.plot(stocks[:, 1], alphas[:, 1], alpha=1, linewidth=2, label=r'$t = \frac{1}{4}$')
        plt.plot(stocks[:, 2], alphas[:, 2], alpha=1, linewidth=2, label=r'$t = \frac{1}{2}$')
        plt.plot(stocks[:, 3], alphas[:, 3], alpha=1, linewidth=2, label=r'$t = \frac{3}{4}$')
        plt.plot(stocks[:, 4], alphas[:, 4], alpha=1, linewidth=2, label=r'$t = 1$')
        
        if american:
            plt.title("Hedging Position for the Underlying Asset in Various Time - American Put")
        else:
            plt.title("Hedging Position for the Underlying Asset in Various Time - European Put")
        plt.xlim(0,25)
        plt.xlabel("Stock Price")
        plt.ylabel(r'$\alpha$')
        plt.legend()
        
        g = plt.figure(2)
        plt.scatter(stocks[:, 0], betas[:, 0], alpha=1, marker='o', linewidth=2, label=r'$t = 0$')
        plt.plot(stocks[:, 1], betas[:, 1], alpha=1, linewidth=2, label=r'$t = \frac{1}{4}$')
        plt.plot(stocks[:, 2], betas[:, 2], alpha=1, linewidth=2, label=r'$t = \frac{1}{2}$')
        plt.plot(stocks[:, 3], betas[:, 3], alpha=1, linewidth=2, label=r'$t = \frac{3}{4}$')
        plt.plot(stocks[:, 4], betas[:, 4], alpha=1, linewidth=2, label=r'$t = 1$')
        
        if american:
            plt.title("Hedging Position for the Numeraire in Various Time - American Put")
        else:
            plt.title("Hedging Position for the Numeraire in Various Time - European Put")  
        plt.xlim(0,25)
        plt.xlabel("Stock Price")
        plt.ylabel(r'$\beta$')
        plt.legend()
        plt.show()
        
    return alphas, betas, stocks


# Q3 part(a) ii
def Q3_iii_boundary_vol(currStockPrice, strikePrice, intRate, vol_list, totSteps, yearsToExp):
    timeStep = yearsToExp / totSteps
    TimeVector = np.arange(0, yearsToExp + timeStep, timeStep)
    fig = plt.figure(1)
    fig.suptitle("Exercise Boundary for Differenct Volatility")
    for vol in vol_list:
        americanOptionTree, euroOptionTree, priceTree, boundary = PutOptionPricer(currStockPrice, strikePrice, intRate, vol, totSteps, yearsToExp)
        label = r'$\sigma=$'+str(vol)
        plt.plot(TimeVector, boundary, label=label)
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()
    
    
def Q3_iii_boundary_r(currStockPrice, strikePrice, intRate_list, vol, totSteps, yearsToExp):
    timeStep = yearsToExp / totSteps
    TimeVector = np.arange(0, yearsToExp + timeStep, timeStep)
    
    fig = plt.figure(1)
    fig.suptitle("Exercise Boundary for Differenct Interest Rate")
    for i in intRate_list:
        americanOptionTree, euroOptionTree, priceTree, boundary = PutOptionPricer(currStockPrice, strikePrice, i, vol, totSteps, yearsToExp)
        label = r'$r=$'+str(i)
        plt.plot(TimeVector, boundary, label=label)
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()
 
    
def Q3_iii_portfolio_vol(currStockPrice, strikePrice, intRate, vol_list, totSteps, yearsToExp):
    fig = plt.figure(1)
    fig.suptitle("Position for Underlying Assets with Differenct Volatility")
    for vol in vol_list:
        americanOptionTree, euroOptionTree, priceTree, boundary = PutOptionPricer(currStockPrice, strikePrice, intRate, vol, totSteps, yearsToExp)
        alphas, betas, stocks = hedgePortfolio(americanOptionTree, priceTree, intRate, vol, totSteps, yearsToExp, True)
        label = r'$\sigma=$'+str(vol)
        plt.plot(stocks[:,2], alphas[:,2], label=label)
    plt.xlim(0,25)
    plt.xlabel("Stock Price")
    plt.ylabel(r'$\alpha$')
    plt.legend()
    
    fig = plt.figure(2)
    fig.suptitle("Position for Numeraire Assets with Differenct Volatility")
    for vol in vol_list:
        americanOptionTree, euroOptionTree, priceTree, boundary = PutOptionPricer(currStockPrice, strikePrice, intRate, vol, totSteps, yearsToExp)
        alphas, betas, stocks = hedgePortfolio(americanOptionTree, priceTree, intRate, vol, totSteps, yearsToExp, True)
        label = r'$\sigma=$'+str(vol)
        plt.plot(stocks[:,2], betas[:,2], label=label)
    plt.xlim(0,25)
    plt.xlabel("Stock Price")
    plt.ylabel(r'$\beta$')
    plt.legend()
    plt.show()
    
    
def Q3_iii_portfolio_r(currStockPrice, strikePrice, intRate_list, vol, totSteps, yearsToExp):
    fig = plt.figure(1)
    fig.suptitle("Position for Underlying Assets with Differenct Interest Rate")
    for i in intRate_list:
        americanOptionTree, euroOptionTree, priceTree, boundary = PutOptionPricer(currStockPrice, strikePrice, i, vol, totSteps, yearsToExp)
        alphas, betas, stocks = hedgePortfolio(americanOptionTree, priceTree, i, vol, totSteps, yearsToExp, True)
        label = r'$r=$'+str(i)
        plt.plot(stocks[:,2], alphas[:,2], label=label)
    plt.xlim(0,25)
    plt.xlabel("Stock Price")
    plt.ylabel(r'$\alpha$')
    plt.legend()
    
    fig = plt.figure(2)
    fig.suptitle("Position for Numeraire Assets with Differenct Interest Rate")
    for i in intRate_list:
        americanOptionTree, euroOptionTree, priceTree, boundary = PutOptionPricer(currStockPrice, strikePrice, i, vol, totSteps, yearsToExp)
        alphas, betas, stocks = hedgePortfolio(americanOptionTree, priceTree, i, vol, totSteps, yearsToExp, True)
        label = r'$r=$'+str(i)
        plt.plot(stocks[:,2], betas[:,2], label=label)
    plt.xlim(0,25)
    plt.xlabel("Stock Price")
    plt.ylabel(r'$\beta$')
    plt.legend()
    plt.show()
         

# Q3 Part(b) i
def putSimulation(optionPrice, currStockPrice, strikePrice, intRate, mu, vol, totSteps, yearsToExp, simulationStep, boundary, plot = False):
    timeStep = yearsToExp/totSteps
    pu = 0.5*(1+ np.sqrt(timeStep)*(mu - intRate - 0.5 * (vol ** 2))/vol)
    
    priceTree = np.full((simulationStep, totSteps+1), np.nan)
    pl_list = np.full(simulationStep, -optionPrice)
    ex_t = np.full(simulationStep, np.nan)
    
    priceTree[:,0] = currStockPrice
    
    for n in range(totSteps):
        U = np.random.rand(simulationStep)
        x = 1*(U<pu) -1*(U>=pu)
        priceTree[:,n+1] = priceTree[:,n] * np.exp(intRate * timeStep + vol * np.sqrt(timeStep) * x)
        for i in range(simulationStep):
            if  (priceTree[i, n+1] <= boundary[n+1]) and np.isnan(ex_t[i]):
                pl_list[i] += (strikePrice-priceTree[i,n+1])*np.exp(-intRate * (n+1) *timeStep)
                ex_t[i] = (n+1) * timeStep
                
    if plot:
        TimeVector = np.linspace(0, totSteps * timeStep, (totSteps + 1))
        fig = plt.figure(1)
        fig.suptitle("Stock Simulation")
        plt.plot(TimeVector,priceTree.T)
        plt.xlabel('Time')
        plt.ylabel('Stock Price')

        # Generate the kernel density estimate
        fig = plt.figure(2)
        fig.suptitle('Kernel Density Estimate of Profit and Loss')
        sns.kdeplot(pl_list,bw_method = 0.05)
        plt.xlabel('Profit and loss')

        # Generate the exercise distribution of function t 
        fig = plt.figure(3)
        fig.suptitle('Kernel Density Estimate of Distribution of Exercise Time')
        sns.kdeplot(ex_t, bw_method = 0.05)
        plt.xlabel('Time to Maturity')
        plt.show()
      
    return priceTree, pl_list, ex_t


# Q3 b - iii
def Q3_b_iii(optionPrice,currStockPrice, strikePrice, intRate, mu, vol_list, totSteps, yearsToExp, simulationStep, early_boundary):
    # Generate plots for multiple volatilities with respect to profit and loss
    data = np.full((10000, 1), np.nan) 
    fig = plt.figure(1)
    for ii in range(len(vol_list)):
        data = putSimulation(optionPrice, currStockPrice, strikePrice, intRate, mu, vol_list[ii], totSteps, yearsToExp, simulationStep, early_boundary)[1]
        sns.kdeplot(np.array(data), bw_method = 0.05, linewidth = 0.7, label = r'$\sigma$=' + str(int(vol_list[ii]*100 )) + '%')    
    fig.suptitle('Kernel Density Estimate of Profit and Loss')
    plt.xlabel('Profit and Loss')
    plt.legend()
    
    # Generate plots for multiple volatilities in terms of exercise time
    fig = plt.figure(2)
    for ii in range(len(vol_list)):
        data = putSimulation(optionPrice, currStockPrice, strikePrice, intRate, mu, vol_list[ii], totSteps, yearsToExp, simulationStep, early_boundary)[2]
        sns.kdeplot(np.array(data), bw_method = 0.05, linewidth = 0.7, label = r'$\sigma$=' + str(int(vol_list[ii]*100 )) + '%')
    fig.suptitle('Kernel Density Estimate of Exercise Time')
    plt.ylim(0,30)
    plt.xlabel('Time to Maturity')
    plt.legend()
    plt.show()

    
if __name__ == "__main__":
    currStockPrice = 10
    strikePrice = 10
    intRate = 0.03
    vol = 0.2
    mu = 0.05
    totSteps = 5000
    yearsToExp = 1
    simulationStep = 10000
    times = [0, 0.25, 0.5, 0.75, 1]
    vol_list = [0.1, 0.15, 0.2, 0.25, 0.3]
    intRate_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    americanOptionTree, euroOptionTree, priceTree, boundary = PutOptionPricer(currStockPrice, strikePrice, intRate, vol, totSteps, yearsToExp)
    
    ''' Exercise Boundary '''
    #pltBoundary(boundary, totSteps, yearsToExp) 
    
    ''' Hedging Portfolio '''
    #alphas, betas, stocks=hedgePortfolio(euroOptionTree, priceTree, intRate, vol, totSteps, yearsToExp, False, True)

    '''Various intRate and vol'''
    #Q3_iii_boundary_vol(currStockPrice, strikePrice, intRate, vol_list, totSteps, yearsToExp)
    #Q3_iii_boundary_r(currStockPrice, strikePrice, intRate_list, vol, totSteps, yearsToExp)
    #Q3_iii_portfolio_vol(currStockPrice, strikePrice, intRate, vol_list, totSteps, yearsToExp)
    #Q3_iii_portfolio_r(currStockPrice, strikePrice, intRate_list, vol, totSteps, yearsToExp)
    
    '''Simulation Distribution'''
    #priceTree, pl_list, ex_t = putSimulation(americanOptionTree[0,0],currStockPrice, strikePrice, intRate, mu, vol, totSteps, yearsToExp, simulationStep, boundary, plot = True)
    
    '''Distributions of profit and loss and exercise time vary in Sigma'''
    Q3_b_iii(americanOptionTree[0,0],currStockPrice, strikePrice, intRate, mu, vol_list, totSteps, yearsToExp, simulationStep, boundary)
