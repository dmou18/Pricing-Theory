import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pdb

class BS():
    def SimStock(Nsims, Nsteps, dt, S_0, mu, sigma):
        BM = np.full((Nsims, Nsteps+1), np.nan)
        sim_paths = np.full((Nsims, Nsteps+1), np.nan)
        
        BM[:,0] = 0
        BM[:, 1:] = np.cumsum(norm.rvs(size= (Nsims, Nsteps), scale=np.sqrt(dt)), axis=1)
        
        sim_paths[:,0] = S_0
        
        for i in range(1, Nsteps+1):
            sim_paths[:,i] = S_0*np.exp((mu - 0.5*np.square(sigma))*dt*i + sigma*BM[:, i])
        
        return sim_paths
    

    def CallPrice(S, T, K, sigma, r):
        
        dp = (np.log(S/K) + (r+0.5*sigma**2)*T)/(np.sqrt(T)*sigma)
        dm = (np.log(S/K) + (r-0.5*sigma**2)*T)/(np.sqrt(T)*sigma)
        
        ans = S*norm.cdf(dp) - K*np.exp(-r*T)*norm.cdf(dm)
        
        ans = np.where(T == 0, np.maximum(S-K, 0), ans)
        
        return ans
    
    
    def PutPrice(S, T, K, sigma, r):
        
        dp = (np.log(S/K) + (r+0.5*sigma**2)*T)/np.sqrt(T)/sigma
        dm = (np.log(S/K) + (r-0.5*sigma**2)*T)/np.sqrt(T)/sigma
        
        ans = K*np.exp(-r*T)*norm.cdf(-dm) - S*norm.cdf(-dp)
        
        ans = np.where(T == 0, np.maximum(K - S, 0), ans)
            
        return ans
    
    
    def CallDelta(S, T, K, sigma, r):
        
        dp = (np.log(S/K) + (r+0.5*sigma**2)*T)/np.sqrt(T)/sigma
        ans = norm.cdf(dp)
        
        ans = np.where(T == 0, np.where(S - K > 0, 1, 0), ans)
            
        return ans
    
    
    def PutDelta(S, T, K, sigma, r):
        ans = BS.CallDelta(S, T, K, sigma, r)-1
        
        ans = np.where(T == 0, np.where(K - S > 0, -1, 0), ans)
        return ans
    
    
    def CallGamma(S, T, K, sigma, r):
        dp = (np.log(S/K) + (r+0.5*sigma**2)*T)/np.sqrt(T)/sigma
        ans = norm.pdf(dp)/(S*sigma*np.sqrt(T))
        
        ans = np.where(T == 0, 0, ans)
            
        return ans
    
    def PutGamma(S, T, K, sigma, r):
        return BS.CallGamma(S, T, K, sigma, r)