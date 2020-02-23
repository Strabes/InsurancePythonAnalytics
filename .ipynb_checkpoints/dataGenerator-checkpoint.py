import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def classDataGenerator(catgVarLvls = [5,8,3],
                       ncont = 3, N = 100000,
                       responseMean = 0.05):

    '''Author: Greg Strabel
    
    This function creates a dataset for classification.
    Parameters:
    catgVarLvls:   The dataset has len(catgVarLvls) categorical variables,
    such that the ith categorical variable has at most catgVarLvls[i] levels.
    ncont: The number of continuous variables
    N: number of observations
    responseMean: mean of the 0-1 response variable
    Returns pandas DataFrame
    '''

    ncat = len(catgVarLvls)

    n = ncat + ncont

    A = np.random.uniform(-1,1,n**2) * np.random.binomial(1,0.5,n**2)
    A = np.triu(A.reshape(n,n),1)

    A = A + np.identity(n)

    A = np.dot(A,A.T)

    sqrtdiagA = np.zeros((n,n))
    np.fill_diagonal(sqrtdiagA, np.sqrt(1/np.diagonal(A)))

    A = np.dot(sqrtdiagA,np.dot(A,sqrtdiagA))

    Z = np.random.multivariate_normal([0]*n,A,size=N)

    df = pd.DataFrame(stats.norm.cdf(Z))
    df['responseMean'] = np.ones(N)

    for i in range(ncat):
        lvls = catgVarLvls[i]    
        cuts = np.random.dirichlet([1]*(lvls + 1)).cumsum()
        cuts = (cuts - cuts.min())/(1-cuts.min())
        df[i] = pd.cut(df[i],cuts,labels = range(lvls))
        factorMap = dict((k, v) for k, v in zip(range(lvls),
                        np.random.uniform(0.75,1.25,lvls)))
        df.responseMean = df.responseMean * df[i].map(factorMap)

    funcDict = {
        0: lambda x: np.exp(np.random.choice([-1,1])*
                     np.random.uniform(0.5,1)*x/x.max()),
        1: lambda x: np.exp(np.random.choice([-1,1])*
                     np.random.uniform(0.5,1)*
                     np.minimum(np.maximum(x/x.max(),0.2),0.8)),
        2: lambda x: np.exp(np.random.choice([-1,1])*
                     np.random.uniform(0.5,1)*
                     np.log(x + 1) / (np.log(x + 1).max())),
        3: lambda x: np.exp(np.random.choice([-1,1])*
                     np.random.uniform(0.5,1)*
                     np.minimum(np.maximum(
                     np.log(x + 1)/(np.log(x + 1).max()),0.2)
                     ,0.8))
                 }
    
    for i in np.arange(ncat,n,dtype = int):
        varType = np.random.choice(['beta','gamma','lognormal'],p=[0.5,.25,.25])
        f = funcDict[np.random.choice(range(len(funcDict)))]
        if varType == 'beta':
            beta_a = np.random.uniform(0.5,2)
            beta_b = np.random.uniform(1,6)
            df[i] = stats.beta.ppf(df[i].values,a = beta_a, b = beta_b)
        elif varType == 'gamma':
            gamma_shape = np.random.uniform(2,10)
            gamma_scale = np.random.uniform(0.5,2)
            df[i] = stats.gamma.ppf(df[i].values,gamma_shape, scale = gamma_scale)
        elif varType == 'lognormal':
            df[i] = stats.lognorm.ppf(df[i].values, s=0.5)
        df.responseMean = df.responseMean * f(df[i].values)
        
# Create interaction terms:

# First for categorical variables:
    nInter = np.maximum(np.array(catgVarLvls).prod() / 20, 10)
    for i in range(nInter):
        interVars = np.random.choice(ncat,size=2,replace=False)
        interVarLvls = [np.random.choice(catgVarLvls[interVars[0]]),
                np.random.choice(catgVarLvls[interVars[1]])]
        interCoef = np.random.uniform(0.75,1.25)
        mask = (df[interVars[0]]==interVarLvls[0]) & \
           (df[interVars[1]]==interVarLvls[1])
        df.loc[mask,'responseMean'] = df.responseMean[mask] * interCoef

# Second for categorical X continuous variables:
    nInter = np.maximum(np.array(catgVarLvls).prod() / 20, 5)
    for i in range(nInter):
        catInterVar = np.random.choice(ncat)
        catInterVarLvl = np.random.choice(catgVarLvls[catInterVar])
        contInterVar = np.random.choice(np.arange(ncat,n,dtype = int))
        mask = (df[catInterVar] == catInterVarLvl)
        f = funcDict[np.random.choice(range(len(funcDict)))]
        df.loc[mask,'responseMean'] = df.responseMean[mask] *\
                            f(df[contInterVar][mask].values)

# Finally, continuous variable interactions:
    nInter = np.maximum(ncont / 3, 3)
    for i in range(nInter):
        interVars = np.random.choice(np.arange(ncat,n,dtype = int),
                                 size=2,replace=False)
        f = funcDict[np.random.choice(range(len(funcDict)))]
        df.responseMean = df.responseMean * \
                  f(df[interVars[0]].values * df[interVars[1]].values)

    df.responseMean = df.responseMean * responseMean / df.responseMean.mean()

    df['response'] = np.random.binomial(1,np.minimum(df.responseMean.values,1))
    
    for i in range(ncat):
        df[i] = df.loc[:,i].apply(lambda x: str(x))

    return df



#plt.figure(frameon=False)
#for i in range(n):
#    plt.subplot(2,3,i+1); ax = plt.gca()
#    df[i].plot(kind='hist', ax = ax,
#        alpha = 0.3, color = 'b', edgecolor = 'b', lw = '3')
#    ax.spines['right'].set_visible(False)
#    ax.spines['top'].set_visible(False)
#    ax.yaxis.set_ticks_position('left')
#    ax.xaxis.set_ticks_position('bottom')
#plt.tight_layout()

#df.responseMean.plot(kind='hist', bins = 100)