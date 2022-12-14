import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pdb
import seaborn as sns
from dynamic_hedging import Dynamic_Hedging 

class Analysis():
    def PlotDeltaHedging(spotPrice, Nsteps, T, delta, bankAccount, lazyDelta, lazyBankAccount):
        timeVector = np.linspace(0, T, Nsteps + 1)
        
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
        
        # #Plot simulation paths for move-based delta
        fig = plt.figure(3)
        fig.suptitle("Move-based Delta Position for Delta Hedging")
        plt.xlabel('Time')
        plt.ylabel(r'$\Delta$')
        plt.plot(timeVector, lazyDelta.T)
        
        #Plot simulation paths for bank account
        fig = plt.figure(4)
        fig.suptitle("Time-based Bank Account for Delta Hedging")
        plt.xlabel('Time')
        plt.ylabel('Bank Account Value')
        plt.plot(timeVector, bankAccount.T)
        
        #Plot simulation paths for move-based bank account
        fig = plt.figure(5)
        fig.suptitle("Move-based Bank Account for Delta Hedging")
        plt.xlabel('Time')
        plt.ylabel('Bank Account Value')
        plt.plot(timeVector, lazyBankAccount.T)
        
        plt.show()
        
        
    def PlotDeltaGammaHedging(spotPrice, Nsteps, T, alpha, gamma, bankAccount, lazyAlpha, lazyGamma, lazyBankAccount):
        timeVector = np.linspace(0, T, Nsteps + 1)
        
        #Plot simulation paths for stock process
        fig = plt.figure(1)
        fig.suptitle("Spot Price Paths")
        plt.xlabel('Time')
        plt.ylabel('Spot Price')
        plt.plot(timeVector, spotPrice.T)
        
        #Plot simulation paths for delta
        fig = plt.figure(2)
        fig.suptitle("Time-based Underlying Asset Position for Delta-Gamma Hedging")
        plt.xlabel('Time')
        plt.ylabel(r'$\alpha$')
        plt.plot(timeVector, alpha.T)
        
        #Plot simulation paths for delta
        fig = plt.figure(3)
        fig.suptitle("Move-based Underlying Asset Position for Delta-Gamma Hedging")
        plt.xlabel('Time')
        plt.ylabel(r'$\alpha$')
        plt.plot(timeVector, lazyAlpha.T)
        
        #Plot simulation paths for gamma
        fig = plt.figure(4)
        fig.suptitle("Time-based Call Option Position for Delta-Gamma Hedging")
        plt.xlabel('Time')
        plt.ylabel(r'$\gamma$')
        plt.plot(timeVector, gamma.T)
        
        #Plot simulation paths for gamma
        fig = plt.figure(5)
        fig.suptitle("Move-based Call Option Position for Delta-Gamma Hedging")
        plt.xlabel('Time')
        plt.ylabel(r'$\gamma$')
        plt.plot(timeVector, lazyGamma.T)
        
        #Plot simulation paths for bank account
        fig = plt.figure(6)
        fig.suptitle("Time-based Bank Account for Delta-Gamma Hedging")
        plt.xlabel('Time')
        plt.ylabel('Bank Account Value')
        plt.plot(timeVector, bankAccount.T)
        
        #Plot simulation paths for bank account
        fig = plt.figure(7)
        fig.suptitle("Move-based Bank Account for Delta-Gamma Hedging")
        plt.xlabel('Time')
        plt.ylabel('Bank Account Value')
        plt.plot(timeVector, lazyBankAccount.T)
        
        plt.show()
        
        
    def deltaPort(spotPrice, delta, bankAccount, settle = False):
        if settle:
             return bankAccount[:,-1]
        else:
            return delta[:,-2]*spotPrice[:,-1] + bankAccount[:,-1]
        
        
    def deltaGammaPort(spotPrice, alpha, gamma, callOption, bankAccount, settle = False):
        if settle:
             return bankAccount[:,-1]
        else:
            return alpha[:,-2]*spotPrice[:,-1] + gamma[:,-2]*callOption[:,-1] + bankAccount[:,-1] 
        
        
    def plotPort(K, S, Portfolio, title, is_call = False):
        price_vector = np.linspace(0, K+150, K+150)
        pl = np.where(K - price_vector > 0,  K - price_vector, 0)
        
        plt.scatter(S, Portfolio, facecolors='none', edgecolors='r')
        plt.plot(price_vector, pl)
        plt.title(title)
        plt.xlabel('Spot Price')
        plt.ylabel('Portfolio Value')
        plt.show()
        
        
    def HistPnL(portfolio, title):
        #plt.hist(portfolio, edgecolor='black', linewidth=1.2)
        sns.distplot(portfolio, bins = 50)
        plt.title(title)
        plt.xlabel('Profit & Loss')
        plt.ylabel('Density')
        # plt.xlim(-2, 2)
        #plt.ylim(0,1.4)
        plt.show()
        
    
    def HistPnL_2 (portfolio1, portfolio2, label1, label2, title):
        sns.distplot(portfolio1, bins = 50, label=label1)
        sns.distplot(portfolio2, bins = 50, label=label2)
        plt.title(title)
        plt.xlabel('Profit & Loss')
        plt.ylabel('Density')
        plt.legend()
        plt.show()
        
        
    def HistTrades(numTrades, title):
        plt.hist(numTrades, bins=20, edgecolor='black', linewidth=1.2)
        plt.title(title)
        plt.xlabel('Unit of Asset Traded')
        plt.ylabel('Frequency')
        plt.show()
        
    
    def CVaR(portfolio, optionPrice, c_level, benchmark, r, T, plot=False):
        VaR = np.quantile(portfolio, c_level)
        CVaR = np.mean(portfolio[portfolio<=VaR])
        premium = (benchmark - CVaR)*np.exp(-r*T)
        adjustedPrice = optionPrice + premium
        
        print(f"The unadjusted option price according to the Delta-Gamma hedging is {optionPrice}")
        print(f"The VaR for the portfolio at level {c_level} is {VaR}")
        print(f"The CVaR for the portfolio at VaR level {c_level} is {CVaR}")
        print(f"The Adjusted put option price so CVaR at VaR level {c_level} is no smaller than {benchmark} is {adjustedPrice}")
        
        if plot:
            ax = sns.distplot(portfolio, bins = 50)
            line = ax.get_lines()[-1]
            x, y = line.get_data()
            x, y = x[x<=VaR], y[x<=VaR]
            ax.fill_between(x, y1=y, alpha=0.5, facecolor='red')
            
            plt.title("Profit and Loss with Conditional Value at Risk")
            plt.xlabel('Profit & Loss')
            plt.ylabel('Density')
            plt.axvline(x=VaR, ls='--', color='purple', label = f"VaR Level")
            plt.axvline(x=benchmark, ls='--', color = 'teal', label=f"Benchmark CVaR Loss")
            plt.axvline(x=CVaR, ls='--', color = 'black', label=f"Conditional VaR")
            
            plt.legend()
            plt.show()
        return CVaR, adjustedPrice
    
    
    def EfficientFrontier(spotPrice, Nsteps, T, dt, K, sigma, r, transCost, bandwidth_list, settle):
        n = len(bandwidth_list)
        mean_list = np.full(n, np.nan)
        std_list = np.full(n, np.nan)
        
        for i in range(n):
            bandwidth = bandwidth_list[i]
            delta, bankAccount, optionPrice, numTrades = Dynamic_Hedging.MoveBasedDeltaHedging(spotPrice, Nsteps, T, dt, K, sigma, r, transCost, bandwidth, settle)
            portfolio =  Analysis.deltaPort(spotPrice, delta, bankAccount, settle)
            mean_list[i] = portfolio.mean()
            std_list[i] = portfolio.std()
        
        fig = plt.figure(1)
        fig.suptitle('Mean Return of Portflio v.s. Bandwidth')
        plt.xlabel("Bandwidth")
        plt.ylabel("Mean P&L of Portfolio")
        plt.plot(bandwidth_list*2, mean_list)        
        
        fig = plt.figure(2)
        fig.suptitle('Standard Deviation of Portflio v.s. Bandwidth')
        plt.xlabel("Bandwidth")
        plt.ylabel("Standard Deviation")
        plt.plot(bandwidth_list*2, std_list)  
        
        fig = plt.figure(3)
        fig.suptitle("Efficient Frontier for Delta Hedging with Various Bandwidth")
        plt.plot(std_list, mean_list)
        plt.xlabel("Standard Deviation of Portfolio")
        plt.ylabel("Mean P&L of Portfolio")
          
        plt.show()
        
        return mean_list, std_list
            
            
    def EfficientFrontier_DG(spotPrice, Nsteps, T1, T2, dt, K, sigma, r, equityTransCost, optTransCost, bandwidth_list, settle):
        n = len(bandwidth_list)
        mean_list = np.full(n, np.nan)
        std_list = np.full(n, np.nan)
        for i in range(n):
            bandwidth = bandwidth_list[i]
            alpha, gamma, callOption, putOption, bankAccount = Dynamic_Hedging.MoveBasedDeltaGammaHedging(spotPrice, Nsteps, T1, T2, dt, K, sigma, r, equityTransCost, optTransCost, bandwidth, settle)
            portfolio = Analysis.deltaGammaPort(spotPrice, alpha, gamma, callOption, bankAccount, settle)
            mean_list[i] = portfolio.mean()
            std_list[i] = portfolio.std()
            
        fig = plt.figure(1)
        fig.suptitle('Mean Return of Portflio v.s. Bandwidth')
        plt.xlabel("Bandwidth")
        plt.ylabel("Mean P&L of Portfolio")
        plt.plot(bandwidth_list*2, mean_list)        
        
        fig = plt.figure(2)
        fig.suptitle('Standard Deviation of Portflio v.s. Bandwidth')
        plt.xlabel("Bandwidth")
        plt.ylabel("Standard Deviation")
        plt.plot(bandwidth_list*2, std_list)  
        
        fig = plt.figure(3)
        fig.suptitle("Efficient Frontier for Delta-Gamma Hedging with Various Bandwidth")
        plt.plot(std_list, mean_list)
        plt.xlabel("Standard Deviation of Portfolio")
        plt.ylabel("Mean P&L of Portfolio")
          
        plt.show()
        
        return mean_list, std_list