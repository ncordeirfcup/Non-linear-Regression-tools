import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
from PIL import ImageTk, Image
#import pymysql
import os
import shutil
import numpy as np
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askopenfilenames
import time
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from cross_validation import cross_validation as cv
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
#from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
#from sklearn.metrics import roc_curve
#from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from rm2 import rm2
from sklearn.metrics import mean_absolute_error,mean_squared_error




initialdir=os.getcwd()
#RF=RandomForestClassifier()

def data1():
    global filename1
    filename1 = askopenfilename(initialdir=initialdir,title = "Select sub-training file")
    firstEntryTabThree.delete(0, END)
    firstEntryTabThree.insert(0, filename1)
    global c_
    c_,d_=os.path.splitext(filename1)
    global file1
    file1 = pd.read_csv(filename1)
    global col1
    col1 = list(file1.head(0))
    
def data2():
    global filename2
    filename2 = askopenfilename(initialdir=initialdir,title = "Select test file")
    secondEntryTabThree.delete(0, END)
    secondEntryTabThree.insert(0, filename2)
    global file2
    file2 = pd.read_csv(filename2)
    
def data3():
    global filename3
    filename3 = askopenfilename(initialdir=initialdir,title = "Select parameter file")
    thirdEntryTabThree_x.delete(0, END)
    thirdEntryTabThree_x.insert(0, filename3)
    global e_
    e_,f_=os.path.splitext(filename3)
    global file3
    file3 = pd.read_csv(filename3)
    
def correlation(X,cthreshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = X.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] > cthreshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in X.columns:
                    del X[colname] # deleting the column from the dataset
    return X   

def variance(X,threshold):
    from sklearn.feature_selection import VarianceThreshold
    sel = VarianceThreshold(threshold=(threshold* (1 - threshold)))
    sel_var=sel.fit_transform(X)
    X=X[X.columns[sel.get_support(indices=True)]]    
    return X

def pretreat(X,cthreshold,vthreshold):
    X=correlation(X,cthreshold)
    X=variance(X,vthreshold)
    return X
    

def selected():
    x=file3.columns.to_list()
    ls,ls2=[],[]
    for i in file3.columns:
        ls.append((list(file3[i])))
    for i in ls:
        ls2.append([x for x in i if x!=0 and str(x)!='nan'])  
    param_grid = dict(zip(x,ls2))
    print(param_grid)
    
    if Criterion.get()==1:
        estimator=RandomForestRegressor(random_state=42)
        rn='RF'
    elif Criterion.get()==2:
        estimator=KNeighborsRegressor()
        rn='KNN'

        
    elif Criterion.get()==4:
        estimator=SVR()
        rn='SVR'
        
    elif Criterion.get()==5:
         estimator=GradientBoostingRegressor(verbose=0, random_state=42)
         rn='GB'
         
    elif Criterion.get()==6:
         estimator=MLPRegressor(max_iter=7000)
         rn='MLP'
         #param_grid ['hidden_layer_sizes']=[(50,50)]
         #param_grid= {"hidden_layer_sizes": [(50, 50)], "activation": ["identity", "logistic", "tanh", "relu"],
                      #'alpha': [0.0001, 0.001, 0.01, 0.1],'learning_rate': ['constant','adaptive', 'invscaling']}
         if int(thirdEntryTabThreer3c1_h.get())!=0:
            param_grid ['hidden_layer_sizes']=[(int(thirdEntryTabThreer3c1_h.get()),)]
         if int(thirdEntryTabThreer3c1_h1.get())!=0:
            param_grid ['hidden_layer_sizes']=[(int(thirdEntryTabThreer3c1_h.get()),int(thirdEntryTabThreer3c1_h1.get()))]
         if int(thirdEntryTabThreer3c1_h2.get())!=0:
            param_grid ['hidden_layer_sizes']=[(int(thirdEntryTabThreer3c1_h.get()),int(thirdEntryTabThreer3c1_h1.get()),int(thirdEntryTabThreer3c1_h2.get()))]
         print(param_grid)
         
    else:
        pass
    rn='g_'+rn
    return estimator,param_grid,rn

   
 
def sol():
    Xtr=file1.iloc[:,2:]
    ytr=file1.iloc[:,1:2]
    ntr=file1.iloc[:,0:1]
    #Xts=file2.iloc[:,2:]
    estimator,param_grid,rn=selected()
    cvg=thirdEntryTabOne.get()
    cvg=int(cvg)
    cvm=forthEntryTabOne.get()
    cvm=int(cvm)
    #param_grid=paramgrid()
    clf = GridSearchCV(cv=cvg, estimator=estimator, param_grid=param_grid, n_jobs=-1)
    cthreshold=float(thirdEntryTabThreer3c1.get())
    vthreshold=float(fourthEntryTabThreer5c1.get())
    Xtr=pretreat(Xtr,cthreshold,vthreshold)
    Xts=file2[Xtr.columns]
    Xtr.to_csv('Pretreat_train_'+str(cthreshold)+'_'+str(vthreshold)+'.csv')
    Xts.to_csv('Pretreat_test_'+str(cthreshold)+'_'+str(vthreshold)+'.csv')
    clf.fit(Xtr,ytr)
    global clfb
    clfb=clf.best_estimator_
    clfb.fit(Xtr,ytr)
    
    
    filer = open(str(c_)+rn+"_tr.txt","w")

    filer.write('The best estimator is: '+'\n')
    filer.write(str(clf.best_estimator_))
    filer.write("\n")
    filer.write((str(cvm)+' fold cross validation statistics are: '+'\n'))
    filer.write("\n")
    writefile2(Xtr,ytr,ntr,clfb,cvm,filer,rn)
    filer.write("\n")
   

def writefile2(Xtr,ytr,ntr,model,cvm,filer,rn):
    cvv=cv(Xtr,ytr,ntr,model,cvm)
    r2,mae,q2lmo,rm2tr,drm2tr,ls=cvv.fit()
    dftr1=pd.concat([ntr,Xtr],axis=1)
    #dftr2=ls.iloc[:,0:2]
    dftr=pd.merge(dftr1,ls.iloc[:,0:3],on=ls.iloc[:,0:1].columns[0],how='left')
    dftr.to_csv(str(c_)+str(rn)+"_trpr.csv",index=False)
    filer.write('R2: '+str(r2)+"\n")
    filer.write(str(cvm)+'-fold cross-validated R2: '+str(q2lmo)+"\n")
    filer.write('Mean absolute error: '+str(mae)+"\n")
    filer.write('Rm2tr '+str(rm2tr)+"\n")
    filer.write('Delta Rm2tr '+str(drm2tr)+"\n")
    if ytr.columns[0] in file2.columns:
       Xts=file2[Xtr.columns]
       nts=file2.iloc[:,0:1]
       yts=file2.iloc[:,1:2]
       ytspr=pd.DataFrame(model.predict(Xts))
       ytspr.columns=['Pred']
       rm2ts,drm2ts=rm2(yts,ytspr).fit()
       tsdf=pd.concat([yts,pd.DataFrame(ytspr)],axis=1)
       tsdf.columns=['Active','Predict']
       tsdf['Aver']=ytr.values.mean()
       tsdf['Aver2']=tsdf['Predict'].mean()
       tsdf['diff']=tsdf['Active']-tsdf['Predict']
       tsdf['diff2']=tsdf['Active']-tsdf['Aver']
       tsdf['diff3']=tsdf['Active']-tsdf['Aver2']
       maets=mean_absolute_error(tsdf['Active'],tsdf['Predict'])
       r2pr=1-((tsdf['diff']**2).sum()/(tsdf['diff2']**2).sum())
       r2pr2=1-((tsdf['diff']**2).sum()/(tsdf['diff3']**2).sum())
       RMSEP=((tsdf['diff']**2).sum()/tsdf.shape[0])**0.5
       dfts=pd.concat([nts,Xts,yts,ytspr],axis=1)
       dfts.to_csv(str(c_)+str(rn)+"_tspr.csv",index=False)
       filer.write("\n")
       filer.write('Test set results: '+"\n")
       filer.write('Number of observations: '+str(yts.shape[0])+"\n")
       filer.write('MAEtest: '+ str(maets)+"\n")
       filer.write('Q2F1/R2Pred: '+ str(r2pr)+"\n")
       filer.write('Q2F2: '+ str(r2pr2)+"\n")
       filer.write('rm2test: '+str(rm2ts)+"\n")
       filer.write('delta rm2test: '+str(drm2ts)+"\n")
       filer.write('RMSEP: '+str(RMSEP)+"\n")
       filer.write("\n")
       
    else:
        Xts=file2.iloc[:,1:]
        nts=file2.iloc[:,0:1]
        ytspr=pd.DataFrame(model.predict(Xts))
        ytspr.columns=['Pred']
        #adts=apdom(Xts[a],Xtr)
        #yadts=adts.fit()
        dfts=pd.concat([nts,Xts,ytspr],axis=1)
        dfts.to_csv(str(c_)+str(rn)+"_scpr.csv",index=False)
    
form = tk.Tk()

form.title("QSAR-Co-X (Module-2)")

form.geometry("650x350")


tab_parent = ttk.Notebook(form)


tab1 = tk.Frame(tab_parent, background='#ffffff')


tab_parent.add(tab1, text="Grid search based non-linear model")


###Tab1#####
    
firstLabelTabThree = tk.Label(tab1, text="Select sub-training set",font=("Helvetica", 12))
firstLabelTabThree.place(x=60,y=10)
firstEntryTabThree = tk.Entry(tab1, width=40)
firstEntryTabThree.place(x=230,y=13)
b3=tk.Button(tab1,text='Browse', command=data1,font=("Helvetica", 10))
b3.place(x=480,y=10)  

secondLabelTabThree = tk.Label(tab1, text="Select test set",font=("Helvetica", 12))
secondLabelTabThree.place(x=120,y=40)
secondEntryTabThree = tk.Entry(tab1,width=40)
secondEntryTabThree.place(x=230,y=43)
b4=tk.Button(tab1,text='Browse', command=data2,font=("Helvetica", 10))
b4.place(x=480,y=40)

thirdLabelTabThree_x = tk.Label(tab1, text="Select parameter file",font=("Helvetica", 12))
thirdLabelTabThree_x.place(x=70,y=70)
thirdEntryTabThree_x = tk.Entry(tab1,width=40)
thirdEntryTabThree_x.place(x=230,y=73)
b4_x=tk.Button(tab1,text='Browse', command=data3,font=("Helvetica", 10))
b4_x.place(x=480,y=70)

Criterion_Label = ttk.Label(tab1, text="Method:",font=("Helvetica", 12))
Criterion = IntVar()
#Criterion.set()
Criterion_RF = ttk.Radiobutton(tab1, text='Random Forest', variable=Criterion, value=1, command=selected)
Criterion_KNN = ttk.Radiobutton(tab1, text='k-Nearest Neighborhood', variable=Criterion, value=2, command=selected)
#Criterion_NB = ttk.Radiobutton(tab1, text='BernoulliNB', variable=Criterion, value=3, command=selected)
Criterion_SVC = ttk.Radiobutton(tab1, text='Support Vector Machine', variable=Criterion, value=4, command=selected)
Criterion_GB = ttk.Radiobutton(tab1, text='Gradient Boosting', variable=Criterion, value=5, command=selected)
Criterion_MLP = ttk.Radiobutton(tab1, text='Multilayer Perception', variable=Criterion, value=6, command=selected)


Criterion_Label.place(x=30,y=100)
Criterion_RF.place(x=100,y=100)
Criterion_KNN.place(x=210,y=100)
#Criterion_NB.place(x=370,y=100)
Criterion_SVC.place(x=370, y=100)
Criterion_GB.place(x=520, y=100)
Criterion_MLP.place(x=130, y=130)

thirdLabelTabThreer2c1_h=Label(tab1, text='Hidden Layers',font=("Helvetica", 10))
v1=IntVar(tab1,value=100)
v2=IntVar(tab1,value=0)
v3=IntVar(tab1,value=0)
thirdLabelTabThreer2c1_h.place(x=270,y=130)
thirdEntryTabThreer3c1_h=Entry(tab1,textvariable=v1,width=5)
thirdEntryTabThreer3c1_h.place(x=370,y=133)
thirdEntryTabThreer3c1_h1=Entry(tab1,textvariable=v2,width=5)
thirdEntryTabThreer3c1_h1.place(x=420,y=133)
thirdEntryTabThreer3c1_h2=Entry(tab1,textvariable=v3,width=5)
thirdEntryTabThreer3c1_h2.place(x=470, y=133)


thirdLabelTabThreer2c1=Label(tab1, text='Correlation cut-off',font=("Helvetica", 12))
thirdLabelTabThreer2c1.place(x=150,y=160)
thirdEntryTabThreer3c1=Entry(tab1)
thirdEntryTabThreer3c1.place(x=300,y=160)

fourthLabelTabThreer4c1=Label(tab1, text='Variance cut-off',font=("Helvetica", 12))
fourthLabelTabThreer4c1.place(x=150,y=190)
fourthEntryTabThreer5c1=Entry(tab1)
fourthEntryTabThreer5c1.place(x=300,y=190)

thirdLabelTabOne=tk.Label(tab1, text="CV for grid search",font=("Helvetica", 12))
thirdLabelTabOne.place(x=150,y=220)
thirdEntryTabOne = tk.Entry(tab1, width=20)
thirdEntryTabOne.place(x=300,y=220)

forthLabelTabOne=tk.Label(tab1, text="CV for model predictability",font=("Helvetica", 12))
forthLabelTabOne.place(x=100,y=250)
forthEntryTabOne = tk.Entry(tab1, width=20)
forthEntryTabOne.place(x=300,y=250)

b2=Button(tab1, text='Generate model', command=sol,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b2.place(x=300,y=280)


tab_parent.pack(expand=1, fill='both')


form.mainloop()