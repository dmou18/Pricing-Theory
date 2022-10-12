# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 21:47:06 2022

@author: zhaiz
@editor: Dixin Mou
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Code for Q3 a
def PutOptionPricer(currStockPrice, strikePrice, intRate, mu, vol, totSteps, yearsToExp):
    timeStep = yearsToExp / totSteps
    u = np.exp(intRate * timeStep + vol * np.sqrt(timeStep))
    # one step random walk (price decreases)
    d = np.exp(intRate * timeStep - vol * np.sqrt(timeStep))
    
    # risk neutral probability of an up move
    pu = (np.exp(intRate * timeStep) -d)/(u-d)
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
        optionValueTree[0:ii, ii] = np.maximum(strikePrice - priceTree[0:ii, ii], optionValueTree[0:ii, ii])
       
    EuropValue = intrinsicTree[0,0]
    AmericanValue = optionValueTree[0,0]
    return payoff, optionValueTree, priceTree, intrinsicTree


# Q3 Part(a) Generate the exercise boundary as a function of t
def PlotExerciseBoundary(currStockPrice, strikePrice, intRate, mu, vol, totSteps, yearsToExp, plot = False):
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
def hedge_node(currStockPrice, strikePrice, intRate, mu, vol, totSteps, yearsToExp):
    timeStep = yearsToExp/totSteps
    payoff, optionValueTree, priceTree, intrinsicTree = PutOptionPricer(currStockPrice, strikePrice, intRate, mu, vol, totSteps, yearsToExp)
    u = np.exp(intRate * timeStep + vol*np.sqrt(timeStep))
    d = np.exp(intRate * timeStep-vol*np.sqrt(timeStep))
    times = np.arange(0, 1.1, 0.25)
    portfolio = np.full((totSteps+1, totSteps+1, 2), np.nan)
    for t in times:
    pass
    
    


if __name__ == "__main__":
    PlotExerciseBoundary(10, 10, 0.02, 0.05, 0.2, 5000, 1, plot)