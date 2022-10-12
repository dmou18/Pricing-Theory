# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 21:47:06 2022

@author: zhaiz
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Code for Part3 Q4
def PutOption(currStockPrice, strikePrice, intRate, mu, vol, totSteps, yearsToExp):
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
    
    # 
    for ii in range(backSteps, 0, -1):
        B = np.exp(intRate * timeStep*ii)
        optionValueTree[0:ii, ii-1] = B*oneStepDiscount *(pu * optionValueTree[0:ii, ii]/B + pd * optionValueTree[1:(ii+1), ii]/B)
        intrinsicTree[0:ii, ii-1] =  B*oneStepDiscount *(pu * intrinsicTree[0:ii, ii]/B + pd * intrinsicTree[1:(ii+1), ii]/B)
        optionValueTree[0:ii, ii] = np.maximum(strikePrice - priceTree[0:ii, ii], optionValueTree[0:ii, ii])
    
    #Calculate the put option price for European style and American style    
    EuropValue = intrinsicTree[0,0]
    AmericanValue = optionValueTree[0,0]
                
    return payoff, optionValueTree, priceTree, intrinsicTree

# Generate the 
TimeVector = np.arange(0, 1+1/5000, 1/5000)
payoff = PutOption(10, 10, 0.02, 0.05, 0.2, 5000, 1)[0]
optionValue = PutOption(10, 10, 0.02, 0.05, 0.2, 5000, 1)[1]
price = PutOption(10, 10, 0.02, 0.05, 0.2, 5000, 1)[2]
priceValue  = PutOption(10, 10, 0.02, 0.05, 0.2, 5000, 1)[2]
intrinsicValue = PutOption(10, 10, 0.02, 0.05, 0.2, 5000, 1)[3]

# Q3 Part(a) Generate the exercise boundary as a function of t
diff = np.subtract(payoff, intrinsicValue)
boundary = np.full((5001, 1), np.nan)
# Trying to figure out if there is a way to optimize
for j in range(0, 5001):
    for i in range(0,5001):
        if diff[i,j] <= 0:
            priceValue[i,j]=np.nan
    boundary[j] = np.nanmax(priceValue[:,j])

fig = plt.figure()
fig.suptitle("Exercise Boundary for American Put Option")
plt.plot(TimeVector, boundary)
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.show()


# Q3 Part(a) ii: hedging strategy

# Q3 Part(a) iii: 
volatility = np.arange(0,1,1/1000)
vol = []
p = []
for i in volatility:
    vol += [i]
    p += [PutOption(10, 10, 0.02, 0.05, i, 5000, 1)[1][0,0]]    
plt.plot(vol,p)
plt.title("Calculated Price VS Volatility")
plt.xlabel("Volatility")
plt.ylabel("Price")
plt.show()
#plot risk-free vs price
risk_free = np.arange(0,0.04,0.0005)
r = []
p = []
for ii in risk_free:
    r += [ii]
    p += [PutOption(10, 10, ii, 0.05, 0.2, 5000, 1)[1][0,0]] 

plt.plot(r,p)
plt.title("Calculated Price VS Risk-Free Rate")
plt.xlabel("Risk-Free Rate")
plt.ylabel("Price")
plt.show() 
# Commentary


# Q3 Part(b) i:
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def Putsimulation(currStockPrice, strikePrice, intRate, mu, vol, totSteps, yearsToExp, simulationStep):
    timeStep = yearsToExp/totSteps
    u = np.exp(vol * np.sqrt(timeStep))
    d = np.exp(-vol * np.sqrt(timeStep))
    pu = (np.exp(intRate * timeStep) - d) / (u - d)
    PriceTree = np.full((simulationStep, totSteps+1), np.nan)
    payoffTree = np.full_like(PriceTree, np.nan)
    OptionValueTree = np.full((simulationStep, totSteps+1), np.nan)
    # PL: profit and loss
    PL = np.full((simulationStep, 1), np.nan) 
    # Exercise time at which we would like to exercise the American put option
    exercise_t = np.full((simulationStep, totSteps+1), np.nan) 
    ex_t = np.full((simulationStep, 1), np.nan)
    
    PriceTree[:,0] = currStockPrice
  
    for j in range(1, totSteps+1):
        # Set the probability of random term
        # Set the random numbers
        random_list = np.random.choice([-1,1],simulationStep, p = [pu,1-pu])
        
        PriceTree[:,j] = PriceTree[:,j-1] * np.exp(random_list*vol * np.sqrt(timeStep))
        OptionValueTree[:,j] = np.maximum(strikePrice - PriceTree[:,j],0)
        
    # Algorithm: Select the time at which we exercise the put option and calculate its present value
    optionPrice = PutOption(10, 10, 0.02, 0, 0.2, 500, 1)[1][0,0]
    intrinsicValue = PutOption(10, 10, 0.02, 0, 0.2, 500, 1)[3]
    for i in range(0, totSteps+1):
        # Select the time at which we want to exercise the option, compare with the early exercise boundary
        # Calculate the profit and loss: PV(payoff) - option price at t = 0 
        payoffTree[:,i] = np.where(PriceTree[:,i] < boundary[i],
                                   (- PriceTree[:,i] + strikePrice)*np.exp(-intRate * i*timeStep) - optionPrice, -optionPrice) 
        exercise_t[:,i] = np.where(PriceTree[:,i] < boundary[i], i, 0) 
        # Find the first exercise price in each path
        if i !=0:
            payoffTree[:,i] = np.where(payoffTree[:,i-1] != -optionPrice,payoffTree[:,i-1],payoffTree[:,i])
            exercise_t[:,i] = np.where(exercise_t[:, i-1] != 0, exercise_t[:,i-1], exercise_t[:,i])
            
            
    # Find the leftmost node on given conditions
    for ii in range(0, simulationStep):
        index = np.where(payoffTree[ii,:] != -optionPrice)
        index2 = np.where(exercise_t[ii,:] != 0)
        if np.array(index).size == 0:
            PL[ii] = -optionPrice
            ex_t[ii] = 0
        else:
            PL[ii] = payoffTree[ii, index[0][0]]
            ex_t[ii] = exercise_t[ii, index[0][0]] * timeStep
        
        
    
    return PriceTree, OptionValueTree, payoffTree, PL, ex_t

price = np.transpose(Putsimulation(10, 10, 0.02, 0.05, 0.2, 5000, 1, 10000)[0])
OptionValue = np.transpose(Putsimulation(10, 10, 0.02, 0.05, 0.2, 5000, 1, 10000)[1])
payoff = Putsimulation(10, 10, 0.02, 0.05, 0.2, 5000, 1, 10000)[2]
TimeVector = np.arange(0, 1+1/5000, 1/5000)
PL = Putsimulation(10, 10, 0.02, 0.05, 0.2, 5000, 1, 10000)[3]
# Simulate 10000 sample paths of the asset
fig = plt.figure()
fig.suptitle("Stock Simulation")
plt.plot(TimeVector,price)
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.show()


# Generate the kernel density estimate
sns.set_style('whitegrid')
sns.kdeplot(np.array(PL[:,0]), bw=0.5)
plt.show()

# Generate the exercise distribution of function t 
# !!! need to be fixed
exercise_time = Putsimulation(10, 10, 0.02, 0.05, 0.2, 5000, 1, 10000)[4]
plt.plot(exercise_time)
# fig = plt.figure()
# fig.suptitle("Stock Simulation")
# plt.plot(TimeVector,price)
# plt.xlabel('Time')
# plt.ylabel('Stock Price')
# plt.show()

# Q3 Part(b) ii:





