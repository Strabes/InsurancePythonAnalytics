# Insurance transactions data generator


import numpy as np
import pandas as pd
from scipy import stats
import time
import dataGeneratorClass
from dataGeneratorClass import dataGenerator
from importlib import reload
reload(dataGeneratorClass)

class insuranceDataGenerator(object):
    '''
    Class used for constructing a dataset of insurance transactions
    
    Parameters
    ----------------
    n : number of insured attributes
    N_pre : number of observations to use in constructing the different
        mapping and scaling transforms
    catg_lvls : list of number of levels for categorical data
    start_dt : minimum effective date
    end_dt : maximum effective date
    nOptCovs : number of optional coverages
    nReqCovs : number of required coverages
    severitySize : average severity across all coverage types
    lClaim : Poisson process arrival rate parameter for claim event
    lEndTermSearch : probability that the insured engages in search at
        the end of a policy term
    lIntermSearch : Poisson process arrival rate parameter for interm
        search event
    lAttributeChange : Poisson process arrival rate parameter for attribute
        change event
    lCoverageChange : Poisson process arrival rate parameter for coverage
        change event
    pCheaperPrem : not currently used
    profitLoad : average profit load for the insurer
    compProfitLoad : average profit load the competitor
    '''
    def __init__(self, n = 6, N_pre = 1000,
                 catg_lvls = [3,5,4],
                 start_dt = '1/1/2005',
                 end_dt = '10/15/2016',
                 nOptCovs = 3,
                 nReqCovs = 3,
                 severitySize = 100000,
                 lClaim = 0.05,
                 lEndTermSearch = 0.3,
                 lIntermSearch = 0.2,
                 lAttributeChange = 0.15,
                 lCoverageChange = 0.15,
                 pCheaperPrem = 0.5,
                 profitLoad = 0.15,
                 compProfitLoad = 0.15):
        self.n = n
        n_beta = int((n-len(catg_lvls))/2)
        n_uniform = n - len(catg_lvls) - n_beta
        self.catg_lvls = catg_lvls
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.nOptCovs = nOptCovs
        self.nReqCovs = nReqCovs
        self.nCovs = nOptCovs + nReqCovs
        self.pCovs = np.random.uniform(size = self.nCovs)
        self.severitySize = severitySize
        self.severitySize_v = severitySize * \
            np.random.uniform(size = self.nCovs)
        self.lClaim = lClaim
        self.lClaim_v = lClaim * self.pCovs / self.pCovs.sum()
        self.lAttributeChange = lAttributeChange
        self.lCoverageChange = lCoverageChange
        self.pCheaperPrem = pCheaperPrem
        self.profitLoad = profitLoad
        self.compProfitLoad = compProfitLoad
        self.dataGenerator = dataGenerator(catg_lvls = self.catg_lvls,
            n_beta = n_beta, n_uniform = n_uniform)
        df, df_imvn = self.dataGenerator.generate(N_pre)
        y = pd.date_range(self.start_dt,self.end_dt)
        self.dateMapper = self.genDateMapper(low=-1,high=1)
        dateTrans = self.dateMapper(np.random.choice(y,df.shape[0]) )
        df_imvn_dt = pd.concat([df_imvn,
                  pd.DataFrame(dateTrans,columns=['date'])],
            axis = 1)
        covlClaimMapper = {}
        pCovMapper = {}
        covSeverityMapper = {}
        profitLoadFactorMapper = {}
        compProfitLoadFactorMapper = {}
        self.endTermSearchMapper = self.dataGenerator.\
                    genSigmoidTransform(df_imvn_dt,
                      width = lEndTermSearch * 1.5,
                      loc = lEndTermSearch)
        self.intermSearchMapper = self.dataGenerator.\
                    genSigmoidTransform(df_imvn_dt,
                      width = lIntermSearch * 1.5,
                      loc = lIntermSearch)
        self.attributeChangeMapper = self.dataGenerator.\
                    genSigmoidTransform(df_imvn_dt,
                      width = lAttributeChange * 1.5,
                      loc = lAttributeChange)
        self.coverageChangeMapper = self.dataGenerator.\
                    genSigmoidTransform(df_imvn_dt,
                      width = lCoverageChange * 1.5,
                      loc = lCoverageChange)


        for i in range(self.nCovs):
            key = 'Coverage: ' + str(i)
            covlClaimMapper[key] = self.dataGenerator.\
                genSigmoidTransform(df_imvn_dt,
                   width = self.lClaim_v[i]*1.5, loc = self.lClaim_v[i])
            if i < self.nReqCovs:
                pCovMapper[key] = lambda x: np.ones(shape = x.shape[0])
            else:
                pCovMapper[key] = self.dataGenerator.\
                genSigmoidTransform(df_imvn_dt,
                   width = self.pCovs[i]*1.5, loc = self.pCovs[i])
            covSeverityMapper[key] = self.dataGenerator.\
                genSigmoidTransform(df_imvn_dt,
                   width = self.severitySize_v[i]*1.5,
                   loc = self.severitySize_v[i])
            profitLoadFactorMapper[key] = self.dataGenerator.\
                genSigmoidTransform(df_imvn_dt,
                   width = self.profitLoad*1.5, loc = self.profitLoad)
            compProfitLoadFactorMapper[key] = self.dataGenerator.\
                genSigmoidTransform(df_imvn_dt,
                   width = self.compProfitLoad*1.5, loc = self.compProfitLoad)
            
        self.covlClaimMapper = covlClaimMapper
        self.pCovMapper = pCovMapper
        self.covSeverityMapper = covSeverityMapper
        self.profitLoadFactorMapper = profitLoadFactorMapper
        self.compProfitLoadFactorMapper = compProfitLoadFactorMapper
        self.lastAccntNumber = int(time.time())
        self.currTransPK = 0
        self.currClaimTransPK = 0

    def genDateMapper(self, cLen = 5, low = 0, high = 1):
        c1_coef = np.random.uniform(-1,1,1).reshape(-1,1)
        c2_coef = np.random.uniform(1,2,1).reshape(-1,1) * \
             np.random.choice([-1,1],size=1).reshape(-1,1)
        def f(date, cLen = cLen, low = low, high = high):
            days = (date - np.datetime64('2000-01-01')).\
               astype('timedelta64[D]').astype(int)
            y = np.sin(2*np.pi*c1_coef*days/(365.25*cLen)) + \
               np.sin(2*np.pi*c2_coef*days/(365.25*cLen))
               #y = 0.25*(y+2)
            y = (0.25*y + 0.5)*(high-low)+low
            return y.reshape(-1)
        return lambda x: f(x, cLen = cLen, low = low, high = high)
    
    def generateAccountTrans(self, nAccounts = 10,
                             start_date = np.datetime64('2000-01-01'),
                             end_date = np.datetime64('today','D'),
                             knowable_columns_only = True):
        df = pd.DataFrame()
        claim_df = pd.DataFrame()
        for i in range(nAccounts):
            policy_number = self.lastAccntNumber + 1
            self.lastAccntNumber = policy_number
            acct_df, acct_claim_df = self.generatePolicyTrans(
                            start_date = start_date,
                            end_date = end_date,
                            policy_number = str(policy_number))
            df = pd.concat([df,acct_df],axis=0)
            claim_df = pd.concat([claim_df,acct_claim_df],axis=0)
        if knowable_columns_only is True:
            df = self.return_vis_data(df)
            claim_df = self.return_vis_data(claim_df)
        df.index = np.arange(self.currTransPK,self.currTransPK + df.shape[0])
        self.currTransPK = self.currTransPK + df.shape[0]
        df.index.name = 'Trans PK'
        try:
            claim_df.index = np.arange(self.currClaimTransPK,
                                   self.currClaimTransPK + claim_df.shape[0])
            self.currClaimTransPK = self.currClaimTransPK + claim_df.shape[0]
            claim_df.index.name = 'Claim Trans PK'
        except:
            print('Error assigning Claim Trans Primary Key')
        return df, claim_df
    
    def generatePolicyTrans(self, start_date = np.datetime64('2000-01-01'),
                             end_date = np.datetime64('today','D'),
                            policy_number = None):
        df, df_imvn_dt = self.dataGenerator.generate_more(1)
        pol_eff_dt = start_date
        pol_exp_dt = self.add_year(start_date) - np.timedelta64(1,'D')
        df_imvn_dt['date'] = self.dateMapper(pol_eff_dt)
        df = pd.merge(df, self.genCovData(df_imvn_dt),
                       left_index = True, right_index = True)
        acct_df = df.copy()
        acct_df['Pol Eff Dt'] = pol_eff_dt
        acct_df['Pol Exp Dt'] = pol_exp_dt
        acct_df['Event'] = 'Issue'
        acct_df['Trans Dt'] = pol_eff_dt
        acct_df['Trans Eff Dt'] = pol_eff_dt
        acct_claim_df = pd.DataFrame()
        cancel_ind = 0
        while cancel_ind == 0 and pol_eff_dt < end_date:
            df, df_imvn_dt, claim_df, cancel_ind, pol_eff_dt = \
            self.genPolicyTermTrans(
                df,df_imvn_dt,pol_eff_dt)
            acct_df = pd.concat([acct_df,df], axis = 0)
            acct_claim_df = pd.concat([acct_claim_df,claim_df], axis = 0)
            df = df[-1:]
            #pol_eff_dt = self.add_year(pol_eff_dt)
            #df_imvn_dt['date'] = self.dateMapper(pol_eff_dt)
        #beta = sum([probs['lclaim: ' + str(i)] for i in range(self.nCovs)])
        acct_df['Trans Number'] = (acct_df.groupby('Pol Eff Dt')
            ['Trans Dt'].rank().astype(int))
        acct_df = acct_df.groupby('Pol Eff Dt').apply(
                lambda x: self.calcTotalWrittenPremium(x))
        acct_df = acct_df.groupby('Pol Eff Dt').apply(
                lambda x: self.calcTransWrittenPremium(x))
        acct_df.loc[:,'Policy Number'] = policy_number
        try:
            acct_claim_df.loc[:,'Policy Number'] = policy_number
        except:
            pass
        return acct_df, acct_claim_df
    
    def genPolicyTermTrans(self, df, df_imvn_dt, dt):
        pol_eff_dt = dt
        pol_exp_dt = self.add_year(dt) - np.timedelta64(1,'D')
        probs, EndTermSearch = self.calcProbabilities(df, df_imvn_dt)
        #print(probs)
        beta = float(sum(probs.values()))
        exp_var = np.random.exponential(scale = beta)
        while int(exp_var*365) == 0:
            exp_var = np.random.exponential(scale = beta)
        cum_exp_var = exp_var
        current_record = df.copy()
        current_record['Pol Eff Dt'] = pol_eff_dt
        current_record['Pol Exp Dt'] = pol_exp_dt
        term_df = pd.DataFrame()
        claim_df = pd.DataFrame()
        cancel_ind = 0
        while cum_exp_var < 1 and cancel_ind == 0:
            trans_dt = pol_eff_dt + np.timedelta64(int(365*cum_exp_var),'D')
            event = np.random.choice(
                    np.array(list(probs.keys())),
                    p=(np.array(list(probs.values()))/sum(list(probs.values()))
                    ).reshape(-1))
            if event == 'Interm Search' and (
                current_record['Total Premium'].values >\
                current_record['Total Comp Premium'].values):
                new_record = current_record.copy()
                if cum_exp_var < 0.25 and np.random.uniform() < 0.5:
                    new_record['Event'] = 'Flat Cancel'
                    new_record['Trans Dt'] = trans_dt
                    new_record['Trans Eff Dt'] = pol_eff_dt
                else:
                    new_record['Event'] = 'Midterm Cancel'
                    new_record['Trans Dt'] = trans_dt
                    new_record['Trans Eff Dt'] = trans_dt
                new_record[['Coverage: ' + str(i) + ' Premium'
                            for i in range(self.nCovs)]] = 0
                            
                cancel_ind = 1
            elif event == 'Attribute Change':
                df_new, df_imvn_dt = self.genAttributeChange(df_imvn_dt)
                df_imvn_dt['date'] = self.dateMapper(pol_eff_dt)
                df = pd.merge(df_new, self.genCovData(df_imvn_dt,
                    coverages = df[['Coverage: ' + str(i)
                    for i in range(self.nCovs)]]),
                    left_index = True, right_index = True)
                probs, EndTermSearch = self.calcProbabilities(df, df_imvn_dt)
                new_record = df.copy()
                new_record['Pol Eff Dt'] = pol_eff_dt
                new_record['Pol Exp Dt'] = pol_exp_dt
                new_record['Event'] = 'Endorsement - Feature Change'
                new_record['Trans Dt'] = trans_dt
                new_record['Trans Eff Dt'] = trans_dt
            elif event == 'Coverage Change':
                df = pd.merge(df[['Feature: ' + str(i)
                    for i in range(self.nCovs)]],
                    self.genCovData(df_imvn_dt),
                    left_index = True, right_index = True)
                probs, EndTermSearch = self.calcProbabilities(df, df_imvn_dt)
                new_record = df.copy()
                new_record['Pol Eff Dt'] = pol_eff_dt
                new_record['Pol Exp Dt'] = pol_exp_dt
                new_record['Event'] = 'Endorsement - Coverage Change'
                new_record['Trans Dt'] = trans_dt
                new_record['Trans Eff Dt'] = trans_dt
            elif 'Claim' in event:
                sev_param = (self.covSeverityMapper
                             ['Coverage:' + event.strip('Claim:')](df_imvn_dt))
                loss = np.random.gamma(sev_param / 2.0, 2.0)
                new_record = None
                loss_record = pd.DataFrame({'Pol Eff Dt':pol_eff_dt,
                                            'Pol Exp Dt': pol_exp_dt,
                                            'Event': 'Claim',
                                            'Coverage':event.strip('Claim: '),
                                            'Trans Dt': trans_dt,
                                            'Loss':loss})
                claim_df = pd.concat([claim_df,loss_record])
            else:
                new_record = None
            beta = float(sum(probs.values()))
            exp_var = np.random.exponential(scale = beta)
            while int(exp_var*365) == 0:
                exp_var = np.random.exponential(scale = beta)
            cum_exp_var = cum_exp_var + exp_var
            term_df = pd.concat([term_df,new_record])

        renewal_term_eff_dt = self.add_year(pol_eff_dt)
        renewal_term_exp_dt = self.add_year(renewal_term_eff_dt) - \
                              np.timedelta64(1,'D')
        
        if cancel_ind == 0:
            df_imvn_dt['date'] = self.dateMapper(renewal_term_eff_dt)
            df = pd.merge(df[['Feature: ' + str(i)
                for i in range(self.n)]],
                self.genCovData(df_imvn_dt,
                coverages = df[['Coverage: ' + str(i)
                for i in range(self.nCovs)]]),
                left_index = True, right_index = True)
            probs, EndTermSearch = self.calcProbabilities(df, df_imvn_dt)
            
            new_record = df.copy()
            new_record['Pol Eff Dt'] = renewal_term_eff_dt
            new_record['Pol Exp Dt'] = renewal_term_exp_dt
            new_record['Trans Dt'] = renewal_term_eff_dt 
            new_record['Trans Eff Dt'] = renewal_term_eff_dt
            
            if EndTermSearch > np.random.uniform() and (
                current_record['Total Premium'].values >\
                current_record['Total Comp Premium'].values):
                new_record['Event'] = 'Renewal Cancel'
                new_record[['Coverage: ' + str(i) + ' Premium'
                            for i in range(self.nCovs)]] = 0
                cancel_ind = 1
            else:
                new_record['Event'] = 'Renewal'
            term_df = pd.concat([term_df,new_record])    
        return term_df, df_imvn_dt, claim_df, cancel_ind, renewal_term_eff_dt


                
    def calcProbabilities(self,df,df_imvn_dt):
        probs = {}
        EndTermSearch = self.endTermSearchMapper(df_imvn_dt)
        probs['Interm Search'] = self.intermSearchMapper(df_imvn_dt)
        probs['Attribute Change'] = self.attributeChangeMapper(df_imvn_dt)
        probs['Coverage Change'] = self.coverageChangeMapper(df_imvn_dt)
        for i in range(self.nCovs):
            #print(df['Coverage: ' + str(i)].values.shape)
            #print(df['Coverage: ' + str(i) + ' lClaim'].values.shape)
            probs['Claim: ' + str(i)] = \
                 df['Coverage: ' + str(i)].values * \
                 df['Coverage: ' + str(i) + ' lClaim'].values
        return probs, EndTermSearch
    
    def add_year(self, x):
        dt = np.datetime_as_string(x).split('-')
        try:
            y = np.datetime64(str(int(dt[0]) + 1) + '-' + dt[1] + '-' + dt[2])
        except:
            y = np.datetime64(str(int(dt[0]) + 1) + '-' + dt[1] + '-' +
                    str(int(dt[2]) - 1))
        return y


    def genAttributeChange(self,df_imvn_dt,attsToChange=None):
        df = df_imvn_dt[['Feature: ' + str(i) for i in range(self.n)]]
        if attsToChange is None:
            n_attsToChange = max(int(df.shape[1]/2),1)
            attsToChange = np.random.choice(df.columns,
                                        size=n_attsToChange,
                                        replace=False).tolist()
        df_imvn_dt2 = pd.DataFrame(index = df.index)
        #print(df_imvn_dt.columns)
        for c in df.columns:
            if c in attsToChange:
                df_imvn_dt2[c] = np.random.normal(size=(df.shape[0],1))
            else:
                df_imvn_dt2[c] = df[c]
        df = pd.DataFrame()
        MVNormQntlDF = pd.DataFrame(stats.norm.cdf(
            np.dot(df_imvn_dt2.values,self.dataGenerator.MVNCoefs.T)),
                columns = df_imvn_dt2.columns)
        df_imvn_dt2['date'] = df_imvn_dt['date']
        for i in self.dataGenerator.catg_cols:
            df[i] = pd.cut(MVNormQntlDF.loc[:,i],
              self.dataGenerator.cat_cuts[i][0],
              labels = self.dataGenerator.cat_cuts[i][1])
        for i in self.dataGenerator.beta_cols:
            df[i] = stats.beta.ppf(MVNormQntlDF.loc[:,i],
              a = self.dataGenerator.cont_beta_params[i][0],
              b = self.dataGenerator.cont_beta_params[i][1])
        for i in self.dataGenerator.uniform_cols:
            df[i] = MVNormQntlDF.loc[:,i]
        return df, df_imvn_dt2
                
        
    def genCovData(self,df_imvn_dt,coverages = None):
        df = pd.DataFrame(index = df_imvn_dt.index)
        for i in range(self.nCovs):
            key = 'Coverage: ' + str(i)
            if coverages is None:
                df[key] = (self.pCovMapper[key](df_imvn_dt) >
                   np.random.uniform(size = df_imvn_dt.shape[0])).astype(int)
            else:
                df[key] = coverages[key]
            df.loc[:,key + ' lClaim'] = self.covlClaimMapper[key](df_imvn_dt)
            df.loc[:,key + ' Severity'] = self.covSeverityMapper[key](df_imvn_dt)
            df.loc[:,key + ' Premium'] = round(
               self.profitLoadFactorMapper[key](df_imvn_dt) * \
               df[key + ' lClaim'] * df[key + ' Severity'] * df[key] , 2)
            df.loc[:,key + ' Comp Premium'] = round(
               self.compProfitLoadFactorMapper[key](df_imvn_dt) * \
               df[key + ' lClaim'] * df[key + ' Severity'] * df[key] , 2)
        df.loc[:,'Total Premium'] = df[
                ['Coverage: ' + str(i) + ' Premium' for i in range(6)]].\
                values.sum(axis = 1)
        df.loc[:,'Total Comp Premium'] = df[
                ['Coverage: ' + str(i) + ' Comp Premium' for i in range(6)]].\
                values.sum(axis = 1)
        return df
    
    def return_vis_data(self, df):
        columns = ['Policy Number', 'Pol Eff Dt', 'Pol Exp Dt',
                   'Trans Number', 'Trans Dt', 'Trans Eff Dt',
                   'Event', 'Total Premium', 'Loss', 'Coverage']
        columns = columns + ['Feature: ' + str(i) for i in range(self.n)]
        columns = columns + ['Coverage: ' + str(i) for i in range(self.nCovs)]
        columns = columns + ['Coverage: ' + str(i) + ' Premium'
                             for i in range(self.nCovs)]
        columns = [i for i in columns if i in df.columns.tolist()]
        return df[columns]
    
    def calcTotalWrittenPremium(self,df):
        df2 = df.iloc[[0],:].copy()
        df_new = df.copy()
        df2.loc[:,'Next Trans Eff Dt'] = df2['Pol Exp Dt'] + \
               np.timedelta64(1,'D')
        coverages = ['Coverage: ' + str(i) + ' Premium' 
                     for i in range(self.nCovs)]
        if df.shape[0] > 1:
            for i in [i for i in df['Trans Number'].values if i > 1]:
                curr_trans = df[df['Trans Number'] == i]
                df2 = df2[df2['Trans Eff Dt'] < \
                                  curr_trans.at[0,'Trans Eff Dt']]
                df2 = pd.concat([df2,curr_trans], axis = 0)
                df2.loc[:,'Next Trans Eff Dt'] = (df2.sort_values('Trans Dt')
                    ['Trans Eff Dt'].shift(-1))
                df2.loc[df2['Next Trans Eff Dt'].isnull(),
                        'Next Trans Eff Dt'] = \
                        curr_trans.at[0,'Pol Exp Dt'] + np.timedelta64(1,'D')
                term_days = ((curr_trans.at[0,'Pol Exp Dt'] -
                              curr_trans.at[0,'Pol Eff Dt']) 
                            / np.timedelta64(1,'D')) + 1
                weights = (((df2['Next Trans Eff Dt'] - df2['Trans Eff Dt']) /
                             np.timedelta64(1,'D')).values) / term_days
                weights = np.tile(weights.reshape(-1,1),(1,self.nCovs))
                df_new.loc[df_new['Trans Number'] == i,coverages] = \
                         sum(weights * df2[coverages].values).round(2)
        df_new.loc[:,'Total Premium'] = df_new[coverages].sum(axis=1).round(2)
        return df_new
    
    def calcTransWrittenPremium(self,df):
        '''Premiums in df must be WP at time of transaction'''
        premiums = ['Coverage: ' + str(i) + ' Premium' 
                     for i in range(self.nCovs)] + ['Total Premium']
        df.loc[:,premiums] = df[premiums] - df[premiums].shift(1).fillna(0)
        return df


if __name__ == '__main__':
    x = insuranceDataGenerator()
    acct_df, claim_df = x.generateAccountTrans(
            start_date = np.datetime64('2010-01-01'))
