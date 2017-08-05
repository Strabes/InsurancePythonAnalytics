# Generate classification and regression datasets
# Author: Greg Strabel

# import modules
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


class dataGenerator(object):
    
    '''
    Class to generate classification and regression datasets.
    Constructs a dataset with categorical and real exogenous variables
    using a Gaussian copula to produce correlations.
    
    Parameters
    --------------
    catg_lvls : list of integers that specify the number of levels for
        categorical variables
    n_beta : number of beta variables
    n_uniform : number of uniform variables
    '''
    
    def __init__(self,catg_lvls = [5,3,4], n_beta = 3, n_uniform = 1):
        self.n_catg = len(catg_lvls)
        self.n_beta = n_beta
        self.n_uniform = n_uniform
        self.n_cont = n_beta + n_uniform
        self.n = self.n_catg + self.n_cont
        self.MVNCoefs, self.A = self._genCopulaParams_(self.n)
        self.featureNames = ['Feature: ' + str(i) for i in range(self.n)]
        self.catg_cols = [self.featureNames[i] for i in 
                          np.arange(0,self.n_catg)]
        self.catg_lvls = dict((k,v) for k,v in zip(self.catg_cols,catg_lvls))
        self.beta_cols = [self.featureNames[i] for i in
                          np.arange(self.n_catg,self.n_catg+n_beta)]
        self.uniform_cols = [self.featureNames[i] for i in
                             np.arange(self.n_catg + n_beta,
                                      self.n_catg + n_beta + n_uniform)]
        self.cont_cols = self.beta_cols + self.uniform_cols
        
    
    def generate(self,N):
        '''
        Generate N observations from the initialized object
        
        Parameters
        --------------
        N : number of observations
        
        Returns
        --------------
        df : pandas DataFrame with formatted exogenous variables
        Z : pandas DataFrame with independent standard Normal
            random variables used to generate df
        '''
        MVNormQntlDF, Z = self._genMVNormQntlDF_(self.n,N)
        MVNormQntlDF.columns = self.featureNames
        Z.columns = self.featureNames
        #MVNormDF = np.dot(Z,self.MVNCoefs.T)
        df = pd.DataFrame(columns = self.featureNames)
        self.cat_cuts = {}
        self.cont_beta_params = {}
        for feature in self.catg_cols:
            df[feature], c, l = self._genCatgVar_(MVNormQntlDF.loc[:,feature],
              lvls = self.catg_lvls[feature])
            self.cat_cuts[feature] = (c,l)
        for feature in self.beta_cols:
            df[feature], a, b = self._genBetaVar_(MVNormQntlDF.loc[:,feature])
            self.cont_beta_params[feature] = (a,b)
        for feature in self.uniform_cols:
            df[feature] = MVNormQntlDF.loc[:,feature]
        return df, Z
    
    def generate_more(self,N):
        '''
        Generate N more observations from the initialized object
        
        Parameters
        --------------
        N : number of observations
        
        Returns
        --------------
        df : pandas DataFrame with format exogenous variables
        Z : pandas DataFrame with independent standard Normal
            random variables used to generate df
        '''
        MVNormQntlDF, Z = self._genMVNormQntlDF_(self.n,N) 
        MVNormQntlDF.columns = self.featureNames
        Z.columns = self.featureNames
        df = pd.DataFrame(columns = self.featureNames)
        for feature in self.catg_cols:
            df[feature] = (pd.cut(MVNormQntlDF.loc[:,feature],
              self.cat_cuts[feature][0],
              labels = self.cat_cuts[feature][1])
                .cat.reorder_categories(
                sorted(self.cat_cuts[feature][1].tolist())
                    ))
        for feature in self.beta_cols:
            df[feature] = stats.beta.ppf(MVNormQntlDF.loc[:,feature],
              a = self.cont_beta_params[feature][0],
              b = self.cont_beta_params[feature][1])
        for feature in self.uniform_cols:
            df[feature] = MVNormQntlDF.loc[:,feature]
        return df, Z
            
    def genSigmoidTransform(self, df_imvn, vars_to_use = 'ALL', **kwargs):
        '''
        Construct and return a function that applies the following mapping to
        df_imvn:
            1. In order to produce interaction effects, select a subset of the
               set of all pairs of columns of df_imvn for interaction terms
            2. Call apply_transform to construct a linear predictor from
               df_imvn and the interaction pairs
            3. Call _sigmoid_scaler_ with key word arguments to produce a
               sigmoid transform with appropriate range
            4. Return a function applying this sigmoid transform with a final
               call to apply_transform
        
        Parameters
        --------------
        df_imvn : pandas DataFrame of independent multivariate Normal R.V.s
        kwargs
        
        Returns
        --------------
        function applying the constructed sigmoid transform
        '''
        if vars_to_use == 'ALL':
            v = df_imvn.columns.tolist()
        else:
            v = vars_to_use
        n_inter = max(int(df_imvn[v].shape[1]/2),1)
        inter_vars = [np.random.choice(df_imvn[v].columns,2).tolist() 
                        for i in np.arange(n_inter)]     
        params = np.random.uniform(-1,1,size = df_imvn[v].shape[1]+n_inter+1)
        lin_pred = self.apply_transform(df_imvn[v],
                    lambda x: x,inter_vars,params)
        f = self._sigmoid_scaler_(lin_pred, **kwargs)
        return lambda x: self.apply_transform(x[v],f,inter_vars,params)
    
    def _genSparseCovMatrix_(self,n):
        A = np.random.uniform(-1,1,n**2) * \
            np.random.binomial(1,0.5,n**2)
        A = np.triu(A.reshape(n,n),1)
        A = A + np.identity(n)
        A = np.dot(A,A.T)
        sqrtdiagA = np.zeros((n,n))
        np.fill_diagonal(sqrtdiagA, np.sqrt(1/np.diagonal(A)))
        A = np.dot(sqrtdiagA,np.dot(A,sqrtdiagA))
        return A
    
    def _genCopulaParams_(self,n):
        '''
        Create two n by n matrices:
        The rows of z have L2 norm of 1.
        The second matrix is positive semi-definite with ones
        on the diagonal
        
        Parameters
        -------------
        n : matrices returned will have shape = (n,n)
        
        Returns
        -------------
        z : (n,n)-shape matrix with rows having L2 norm of 1.
        z * z.T
        '''
        z = np.random.normal(size = (n,n))
        z = z/np.linalg.norm(z,axis=1).reshape(n,1)
        return z, np.dot(z,z.T)
        
    def _genMVNormQntlDF_(self, n, N):
        '''
        Construct and return two pandas DataFrames.
        
        Parameters
        -------------
        n : number of columns
        N : number of rows
        
        Returns
        -------------
        df : N-by-n pandas DataFrame of Gaussian copula observations
        Z : N-by-n pandas DataFrame of underlying independent multivariate
            Normal R.V.s
        '''
        Z = np.random.multivariate_normal([0]*n,np.eye(n),size=N)
        df = pd.DataFrame(stats.norm.cdf(np.dot(Z,self.MVNCoefs.T)))
        return df, pd.DataFrame(Z)
        
    def _genBetaVar_(self,X):
        '''
        Transform uniform random variables to beta random variables
        using the inverse CDF. Return beta variables and beta parameters
        
        Parameters
        -------------
        X : 1d array of values in [0,1]
        
        Returns
        -------------
        Y : 1d array of inverse beta CDF transformed variates
        beta_a : a parameter for beta distribution
        beta_b : b parameter for beta distribution
        '''
        beta_a = np.random.uniform(0.5,2)
        beta_b = np.random.uniform(1,6)
        Y = stats.beta.ppf(X, a = beta_a, b = beta_b)
        return Y, beta_a, beta_b
		
    def _genCatgVar_(self, X, lvls = 5):
        '''
        Transform uniform[0,1] random variables to categorical
        random variable using cumulative dirichlet cuts.
        
        Parameters
        -------------
        X : 1d uniform[0,1] random variables
        lvls : integer number of levels for categorical variable
        
        Returns
        -------------
        Series of categorical variables
        Dirichlet cut points
        Labels for corresponding categorical variables
        '''
        cuts = np.random.dirichlet([1]*(lvls + 1)).cumsum()
        cuts = (cuts - cuts.min())/(1-cuts.min())
        labels = np.random.permutation(lvls)
        Y = pd.cut(X, cuts, labels = labels)
        Y = Y.cat.reorder_categories(
                sorted(Y.cat.categories.tolist()))
        return Y, cuts, labels
    
    def _sigmoid_scaler_(self, X, y_high = 0.75,
                            y_low = 0.25, x_high = None,
                            x_low = None, width = 1, loc = 0.5):
        '''
        Create a scaled and centered version of a sigmoid activation
        function.
        First, solve for parameters of sigmoid activation function that map:
        x_high --> y_high
        x_low --> y_low
        Second, center and scale the resulting function to have range:
        (loc - width/2, loc + width/2)
        
        Parameters
        ---------------
        X : 1d vector of numPy floats
        x_low : value to scale to [width * (y_low - 1/2) + loc]
        x_high : value to scale to [width * (y_high - 1/2) + loc] 
        y_low : x_low scales to [width * (y_low - 1/2) + loc]
        y_high : x_high scales to [width * (y_high - 1/2) + loc]
        '''
        if x_high is None:
            x_high = float(np.percentile(X,75))
        if x_low is None:
            x_low = float(np.percentile(X,25))
        a = np.log((y_high*(1-y_low))/(y_low*(1-y_high))) / (x_high - x_low)
        b = -a*x_high - np.log(1/y_high - 1)
        return lambda x: width / (np.exp(-a*x-b) + 1) - 0.5 * width + loc
        
    def apply_transform(self,df_imvn,f,inter_vars,params):
        '''
        Apply the function f to a linear combination of the correlated
        multivariate normal R.V.s constructed from df_imvn * self.MVNCoefs.T
        and the product of some of these terms as specified in inter_vars
        
        Parameters
        -------------
        df_imvn : pandas DataFrame of independent multivariate standard
            normal random variables
        f : function to be applied
        inter_vars : list of lists of names of interaction combinations
        params : parameters for the linear transformation of the union
            of df_imvn * self.MVNCoefs and the interaction terms
        
        Returns
        -------------
        pandas Series
        '''
        p = np.eye(df_imvn.shape[1])
        p[:self.MVNCoefs.T.shape[0],
          :self.MVNCoefs.T.shape[1]] = self.MVNCoefs.T
        d = pd.DataFrame(np.dot(df_imvn.values,p),
                         columns = df_imvn.columns)
        inter_values = np.zeros((d.shape[0],len(inter_vars)))
        for i in range(len(inter_vars)):
            inter_values[:,i] = d[inter_vars[i][0]].values *\
                        d[inter_vars[i][1]].values
        lin_pred = np.dot(
                np.concatenate([np.ones((d.shape[0],1)),
                d.values, inter_values], axis = 1),
                params)
        return f(lin_pred)

    
    def genBernoulliVariates(self,df_imvn,f):
        '''
        Generate 1d array of Bernoulli R.V.s with length = df.shape[0]
        where Pr{1} = f(df)
        
        Parameters
        -------------
        df : pandas DataFrame
        f : function applied to rows of df with codomain = [0,1]
        
        Returns
        -------------
        1d array of Bernoulli variates
        '''
        return np.random.binomial(1,f(df_imvn))

        
    
    
    


def plotData(df,response = None, n_cols = None):
    ncuts = 10
    var_cols = df.columns.tolist()
    if response is not None:
        if type(response) == str:
            response = [response]
        for r in response:
            var_cols.remove(r)
        colors = ['orange','b','y','g','c','m']
        colors = (colors * np.ceil(len(response)/len(colors)).
                  astype(int))[0:len(response)]
    n_vars = len(var_cols)
    if n_cols is None:
        n_cols = int(np.sqrt(n_vars))
    n_rows = int(np.ceil(n_vars / n_cols))
    mlpFig, axes = plt.subplots(n_rows,n_cols,figsize = (13,8))
    catg_vars = df.columns[df.dtypes == 'category'].tolist()
    float_vars = df.columns[df.dtypes == 'float'].tolist()
    for i,v in enumerate(var_cols):
        ax = axes.ravel()[i]
        if v in catg_vars:
            df[v].value_counts().sort_index().\
              plot(kind = 'bar',ax=ax, color = 'xkcd:light grey')
            if response is not None:
                axt = ax.twinx()
                for r,c in zip(response,colors):
                    df[r].groupby(df[v]).mean().sort_index().\
                      plot(ax = axt, color = c)
            plt.xlim([-1,len(df[v].cat.categories)])
        elif v in float_vars:
            cuts, bins = pd.cut(df[v],ncuts,retbins = True)
            cuts.value_counts().sort_index().plot(kind = 'bar',
                  ax=ax, color = 'xkcd:light grey')
            if response is not None:
                axt = ax.twinx()
                for r,c in zip(response,colors):
                    df[r].groupby(cuts).mean().\
                      sort_index().plot(ax = axt,
                          color = c)
            plt.xlim([-1,10])
            ticklabels = [round((bins[j+1]+bins[j])/2,2) \
                                 for j in range(len(bins)-1)]
            ax.set_xticklabels(ticklabels)
        ax.set_title(v)
            #axes.ravel()[i].scatter(df[v],df[response])
    for i,ax in enumerate(axes.flatten()):
        if i >= len(var_cols):
            ax.axis('off')
        else:
            for tk in ax.get_xticklabels():
                tk.set_visible(True); tk.set_rotation(45)
    if response is not None:
        mlpFig.legend(plt.gca().lines,response,'lower center',
                      ncol = n_cols, fontsize = 'large'#,
                      #bbox_to_anchor = (0.5,0.95)
                      )
    plt.tight_layout(rect = [0,0.1,1,1])
    mlpFig.show()
    
if __name__ == '__main__':
    x = dataGenerator()
    df1, df2 = x.generate(1000)
    f = x.genSigmoidTransform(df2)
    df = pd.concat([df1,pd.DataFrame(f(df2),columns=['Response'])],axis=1)
    plotData(df, response = 'Response').show()