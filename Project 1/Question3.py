from pickle import FALSE
import numpy as np
import matplotlib.pyplot as plt

def CRROptionWGreeks(currStockPrice, strikePrice, intRate, mu, vol, totSteps, yearsToExp, optionType, american):
  # calculate the number of time steps that we require
  timeStep = yearsToExp / totSteps
    
  # one step random walk (price increases)
  u = np.exp(intRate * timeStep + vol * np.sqrt(timeStep))
  # one step random walk (price decreases)
  d = np.exp(intRate * timeStep - vol * np.sqrt(timeStep))

  # risk neutral probability of an up move
  pu = 0.5*(1 + (mu - intRate - 0.5*(vol**2))*np.sqrt(timeStep)/vol)
  # risk neutral probability of a down move
  pd = 1 - pu

  # Tree is evaluated in two passes
  # In the first pass the probability weighted value of the stock price is calculated at each node
  # In the second pass the value of the option is calculated for each node given the stock price, backwards from the final time step
  # Note that the tree is recombinant, i.e. u+d = d+u

  # First we need an empty matrix big enough to hold all the calculated prices
  priceTree = np.full((totSteps+1, totSteps+1), np.nan) # matrix filled with NaN == missing values
  # Note that this tree is approx twice as big as needed because we only need one side of diagonal
  # We use the top diagonal for efficiency

  # Initialize with the current stock price, then loop through all the steps
  priceTree[0, 0] = currStockPrice

  for ii in range(1, totSteps+1):
    # vector calculation of all the up steps (show how the indexing works on line)
    priceTree[0:ii, ii] = priceTree[0:ii, (ii-1)] * u

    # The diagonal will hold the series of realizations that is always down
    # this is a scalar calculation
    priceTree[ii, ii] = priceTree[(ii-1), (ii-1)] * d

    #print("\n", priceTree)

  # Now we can calculate the value of the option at each node
  # We need a matrix to hold the option values that is the same size as the price tree
  # Note that size returns a vector of dimensions [r, c] which is why it can be passed as dimensions to nan
  optionValueTree = np.full_like(priceTree, np.nan)
  strategyTree = np.full_like(priceTree, np.nan)

  # First we calculate the terminal value
  if optionType == "CALL":
    optionValueTree[:, -1] = np.maximum(0, priceTree[:, -1] - strikePrice)
    # note the handy matrix shortcut syntax & max applied elementwise
  elif optionType == "PUT":
    optionValueTree[:, -1] = np.maximum(0, strikePrice - priceTree[:, -1])
  else:
    # wherever possible, check that the input parameters are valid and raise an exception if not
    raise ValueError("Only CALL and PUT option types are supported")
  
  if not american:
    inds = np.triu_indices(len(strategyTree))
    strategyTree[inds] = 0
    strategyTree[:, -1] = np.where(optionValueTree[:,-1] > 0, 1, 0) 
  else :
    strategyTree[:,-1] = np.where(optionValueTree[:,-1] > 0, 1, 0) 

  oneStepDiscount = np.exp(-intRate * timeStep)  # discount rate for one step

  # Now we step backwards to calculate the probability weighted option value at every previous node
  # How many backwards steps?
  backSteps = priceTree.shape[1] - 1  # notice the shape function -> 1 is # of columns, which is last index + 1

  for ii in range(backSteps, 0, -1):
    optionValueTree[0:ii, ii-1] = \
      oneStepDiscount * \
      (pu * optionValueTree[0:ii, ii] \
      + pd * optionValueTree[1:(ii+1), ii])

    #if the option is american then you can convert at anytime, so the option value can never be less than the intrinsic value
    if american:
      if optionType == "CALL":
        optionValueTree[0:ii, ii] = np.maximum(priceTree[0:ii, ii] - strikePrice, optionValueTree[0:ii, ii])
        strategyTree[0:ii, ii] = np.where(((priceTree[0:ii, ii] - strikePrice) - optionValueTree[0:ii, ii]) >= 0, 1, 0)
      else:
        optionValueTree[0:ii, ii] = np.maximum(strikePrice - priceTree[0:ii, ii], optionValueTree[0:ii, ii])
        strategyTree[0:ii, ii] = np.where((optionValueTree[0:ii, ii] - (priceTree[0:ii, ii] - strikePrice))  >= 0, 1, 0)

  # After all that, the current price of the option will be in the first element of the optionValueTree
  optionPrice = optionValueTree[0, 0]

  return optionPrice, optionValueTree, strategyTree


if __name__ == "__main__":
  try:
    crr = CRROptionWGreeks(10, 10, 0.02, 0.05, 0.2, 500, 1, "PUT", True)
  except ValueError as e:
    print("Type error: " + str(e))
  except Exception as e:
    print("Unknown error: " + str(e))