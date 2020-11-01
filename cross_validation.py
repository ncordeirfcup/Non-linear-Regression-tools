from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef,recall_score,roc_auc_score,roc_curve,confusion_matrix
import numpy as np
import math
from sklearn.model_selection import KFold  #For K-fold cross validation\n
import random
import copy
import math
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from numpy import ndarray
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.metrics import mean_absolute_error,mean_squared_error
from rm2 import rm2
from sklearn.metrics import r2_score

class cross_validation:
      def __init__(self,X_data,y_data, n, model,cv):
            self.n=n
            self.X_data=X_data
            self.y_data=y_data
            self.cv=cv
            self.model=model
        
      def fit(self):
         kf = KFold(n_splits=self.cv, random_state=None, shuffle=False)
         ls2=[]
         for train, test in kf.split(self.X_data):
            train_predictors = (self.X_data.iloc[train,:])
            train_target = self.y_data.iloc[train]
            self.model.fit(train_predictors,train_target)
            tspred = pd.DataFrame(self.model.predict(self.X_data.iloc[test,:]))
            ls=pd.concat([self.n.iloc[test].reset_index(drop=True),self.y_data.iloc[test].reset_index(drop=True),tspred],axis=1)
            ls2.append(ls)
            a=pd.concat(ls2,axis=0)
         a.columns=[self.n.columns[0],'Active','Predict']
         r2=r2_score(a['Active'],a['Predict'])
         mae=mean_absolute_error(a['Active'],a['Predict'])
         a['Del_Res']=a['Active']-a['Predict']
         a['Mean']=a['Active'].mean()
         a['nsum']=a['Active']-a['Mean']
         q2lmo=1-((a['Del_Res']**2).sum()/(a['nsum']**2).sum())
         rm2tr,drm2tr=rm2(a.iloc[:,1:2],a.iloc[:,2:3]).fit()
         return r2,mae,q2lmo,rm2tr,drm2tr,a
         