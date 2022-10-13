# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 21:47:06 2022

@author: zhaiz, Dixin Mou
"""
from cmath import nan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Code for Q3 a
def PutOptionPricer(currStockPrice, strikePrice, intRate, mu, vol, totSteps, yearsToExp):
    timeStep = yearsToExp / totSteps
    u = np.exp(vol * np.sqrt(timeStep))
    # one step random walk (price decreases)
    d = np.exp(- vol * np.sqrt(timeStep))
    
    # risk neutral probability of an up move
    pu = (1 -d)/(u-d)
    # risk neutral probability of a down move
    pd = 1 - pu
    
    priceTree = np.full((totSteps+1, totSteps+1), np.nan)
    intrinsicTree = np.full((totSteps+1, totSteps+1), np.nan)
    
    priceTree[0, 0] = currStockPrice

    for ii in range(1, totSteps+1):
        priceTree[0:ii, ii] = priceTree[0:ii, (ii-1)] * u
        priceTree[ii, ii] = priceTree[(ii-1), (ii-1)] * d

    optionValueTree = np.full_like(priceTree, np.nan)
    optionValueTree[:, -1] = np.maximum(0, strikePrice - priceTree[:, -1])
    intrinsicTree[:, -1] = np.maximum(0, strikePrice - priceTree[:, -1])
    payoff = np.full_like(priceTree, np.nan)
    payoff[:,:] = np.maximum(0, strikePrice - priceTree[:,:])

    oneStepDiscount = np.exp(-intRate * timeStep) 
    backSteps = priceTree.shape[1] - 1
    
    for ii in range(backSteps, 0, -1):
        B = np.exp(intRate * timeStep*ii)
        optionValueTree[0:ii, ii-1] = B*oneStepDiscount *(pu * optionValueTree[0:ii, ii]/B + pd * optionValueTree[1:(ii+1), ii]/B)
        intrinsicTree[0:ii, ii-1] =  B*oneStepDiscount *(pu * intrinsicTree[0:ii, ii]/B + pd * intrinsicTree[1:(ii+1), ii]/B)
        optionValueTree[0:ii+1, ii] = np.maximum(strikePrice - priceTree[0:ii+1, ii], optionValueTree[0:ii+1, ii])
       
    EuropValue = intrinsicTree[0,0]
    AmericanValue = optionValueTree[0,0]
    return payoff, optionValueTree, priceTree, intrinsicTree


# Q3 Part(a) Generate the exercise boundary as a function of t
def exerciseBoundary(currStockPrice, strikePrice, intRate, mu, vol, totSteps, yearsToExp, plot = False):
    timeStep = yearsToExp / totSteps
    TimeVector = np.arange(0, yearsToExp+timeStep, timeStep)
    payoff, optionValueTree, priceTree, intrinsicTree = PutOptionPricer(currStockPrice, strikePrice, intRate, mu, vol, totSteps, yearsToExp)
    
    diff = np.subtract(payoff, intrinsicTree)
    boundary = np.full((1+totSteps, 1), np.nan)
    
    for i in range(0, 1+totSteps):
        inds = np.argwhere(diff[:,i] > 0)
        if np.size(inds) > 0:
            boundary[i] = priceTree[inds[0], i]

    if plot:
        fig = plt.figure()
        fig.suptitle("Exercise Boundary for American Put Option")
        plt.plot(TimeVector, boundary)
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.show()
    return TimeVector, boundary
   
    
# Q3 Part(a) ii: hedging strategy
def hedgePortfolio(currStockPrice, strikePrice, intRate, mu, vol, totSteps, yearsToExp, plot = False):
    timeStep = yearsToExp/totSteps
    payoff, optionValueTree, priceTree, intrinsicTree = PutOptionPricer(currStockPrice, strikePrice, intRate, mu, vol, totSteps, yearsToExp)
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
            alpha = np.where(optionValueTree[:, ind] > 0, -1, 0)
            beta = (optionValueTree[:, ind] - (S*alpha))/np.exp(intRate * timeStep * ind)
        else:
            Cu = optionValueTree[0:ind+1, ind+1]
            Cd = optionValueTree[1:ind+2, ind+1]
            alpha = (Cu-Cd)/(S*(u-d))
            beta = (Cu - alpha*S*u)/np.exp(intRate * timeStep * ind)
        
        alphas[0:ind + 1,i] = alpha
        betas[0:ind + 1,i] = beta
        stocks[0 : ind + 1, i] = S
         
    inds = np.argwhere(stocks <= 20)
    portfolio_0 = []
    portfolio_1 = []
    portfolio_2 = []
    portfolio_3 = []
    portfolio_4 = []
    
    for i in inds:
        x = i[0]
        y = i[1]
        if y == 0:
            portfolio_0.append([stocks[x,y], alphas[x,y], betas[x,y]])
        elif y == 1:
            portfolio_1.append([stocks[x,y], alphas[x,y], betas[x,y]])
        elif y == 2:
            portfolio_2.append([stocks[x,y], alphas[x,y], betas[x,y]])
        elif y == 3:
            portfolio_3.append([stocks[x,y], alphas[x,y], betas[x,y]])
        elif y == 4:
            portfolio_4.append([stocks[x,y], alphas[x,y], betas[x,y]])
            
    portfolio_0 = np.reshape(portfolio_0, (-1,3))
    portfolio_1 = np.reshape(portfolio_1, (-1,3))
    portfolio_2 = np.reshape(portfolio_2, (-1,3))
    portfolio_3 = np.reshape(portfolio_3, (-1,3))
    portfolio_4 = np.reshape(portfolio_4, (-1,3))
    
    if plot:
        f = plt.figure(1)
        plt.scatter(portfolio_0[:, 0], portfolio_0[:, 1], alpha=1, marker='o', linewidth=3, label=r'$t = 0$')
        plt.plot(portfolio_1[:, 0], portfolio_1[:, 1], alpha=1, linewidth=2, label=r'$t = \frac{1}{4}$')
        plt.plot(portfolio_2[:, 0], portfolio_2[:, 1], alpha=1, linewidth=2, label=r'$t = \frac{1}{2}$')
        plt.plot(portfolio_3[:, 0], portfolio_3[:, 1], alpha=1, linewidth=2, label=r'$t = \frac{3}{4}$')
        plt.plot(portfolio_4[:, 0], portfolio_4[:, 1], alpha=1, linewidth=2, label=r'$t = 1$')
        
        plt.title("Hedging Position for the Underlying Asset in Various Time")
        plt.xlabel("Stock Price")
        plt.ylabel(r'$\alpha$')
        plt.legend()
        
        g = plt.figure(2)
        plt.scatter(portfolio_0[:, 0], portfolio_0[:, 2], alpha=1, marker='o', linewidth=3, label=r'$t = 0$')
        plt.plot(portfolio_1[:, 0], portfolio_1[:, 2], alpha=1, linewidth=2, label=r'$t = \frac{1}{4}$')
        plt.plot(portfolio_2[:, 0], portfolio_2[:, 2], alpha=1, linewidth=2, label=r'$t = \frac{1}{2}$')
        plt.plot(portfolio_3[:, 0], portfolio_3[:, 2], alpha=1, linewidth=2, label=r'$t = \frac{3}{4}$')
        plt.plot(portfolio_4[:, 0], portfolio_4[:, 2], alpha=1, linewidth=2, label=r'$t = 1$')
        
        plt.title("Hedging Position for the Numeraire in Various Time")
        plt.xlabel("Stock Price")
        plt.ylabel(r'$\beta$')
        plt.legend()
        plt.show()
        
    return portfolio_0, portfolio_1, portfolio_2, portfolio_3, portfolio_4
    
    
if __name__ == "__main__":
    # exerciseBoundary(10, 10, 0.02, 0.05, 0.2, 5000, 1, True)
    hedgePortfolio(10, 10, 0.02, 0.05, 0.2, 5000, 1, True)
