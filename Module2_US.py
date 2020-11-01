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
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot
from rm2 import rm2
from sklearn.metrics import mean_absolute_error,mean_squared_error



initialdir=os.getcwd()


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
    filename3 = askopenfilename(initialdir=initialdir,title = "Select validation file")
    fifthEntryTabOne.delete(0, END)
    fifthEntryTabOne.insert(0, filename3)
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

 
def sol():
    Xtr=file1.iloc[:,2:]
    ytr=file1.iloc[:,1:2]
    ntr=file1.iloc[:,0:1]
    Xts=file2.iloc[:,2:]
    estimator,rn=Selected()
    cvm=forthEntryTabOne.get()
    cvm=int(cvm)
    cthreshold=float(thirdEntryTabThreer3c1.get())
    vthreshold=float(fourthEntryTabThreer5c1.get())
    Xtr=pretreat(Xtr,cthreshold,vthreshold)
    Xts=Xts[Xtr.columns]
    Xtr.to_csv('Pretreat_train_'+str(cthreshold)+'_'+str(vthreshold)+'.csv')
    Xts.to_csv('Pretreat_test_'+str(cthreshold)+'_'+str(vthreshold)+'.csv')
    global clf
    clf = estimator
    
    clf.fit(Xtr,ytr)

    clf.score(Xtr,ytr)
   
    filer = open(str(c_)+rn+"_tr.txt","w")
    filer.write('The estimator is: '+'\n')
    filer.write(str(clf))
    filer.write("\n")
    filer.write((str(cvm)+' fold cross validation statistics are: '+'\n'))
    filer.write("\n")
    writefile2(Xtr,ytr,ntr,clf,cvm,filer,rn)
    filer.write("\n")
        
    global pyplot 
    pyplot.figure(figsize=(15,10))
    

def writefile2(Xtr,ytr,ntr,model,cvm,filer,rn):
    cvv=cv(Xtr,ytr,ntr,model,cvm)
    r2,mae,q2lmo,rm2tr,drm2tr,ls=cvv.fit()
    dftr1=pd.concat([ntr,Xtr],axis=1)
    #dftr2=ls.iloc[:,0:2]
    dftr=pd.merge(dftr1,ls.iloc[:,0:3],on=ls.iloc[:,0:1].columns[0],how='left')
    dftr.to_csv(str(c_)+str(rn)+"_trpr.csv",index=False)
    filer.write('R2: '+str(r2)+"\n")
    filer.write('Mean absolute error: '+str(mae)+"\n")
    filer.write(str(cvm)+'-fold cross-validated R2: '+str(q2lmo)+"\n")
    filer.write('Rm2LOO '+str(rm2tr)+"\n")
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
        ytspr=pd.DataFrame(reg.predict(Xts[a]))
        ytspr.columns=['Pred']
        adts=apdom(Xts[a],Xtr[a])
        yadts=adts.fit()
        dfts=pd.concat([nts,Xts[a],ytspr,yadts],axis=1)
        name2="Test_prediction.csv"
        dfts.to_csv(os.path.join(dct,name2),index=False)
        #dfts.to_csv(str(c_)+str(rn)+"_sfslda_scpr.csv",index=False)

def Selected():
    Xtr=file1.iloc[:,2:]
    ytr=file1.iloc[:,1:2]
    ntr=file1.iloc[:,0:1]

    if var1.get():
       wg=Criterion2.get()
       alg=Criterion3.get()
       rn='KNN'
       estimator=KNeighborsRegressor(n_neighbors=int(E1T1.get()),weights=wg, algorithm=alg)
     
    elif var3.get():
         rn='SVR'
         estimator=SVR(C=float(E3T1.get()),gamma=float(E4T1.get()),kernel=Criterion5.get())
         #clf3.fit(Xtr,ytr)
         #print(clf3.score(Xtr,ytr))
    elif var4.get():
         alpha=float(E5T1.get())
         rn='MLP'
         estimator=MLPRegressor(hidden_layer_sizes=(50, 50),alpha=alpha,activation=Criterion6.get(),solver=Criterion7.get(), learning_rate=Criterion8.get())
         #clf4.fit(Xtr,ytr)
         #print(clf4.score(Xtr,ytr))
    elif var5.get():
         rn='RFR'
         if Criterion25.get()=='none':
            md=None
         elif Criterion25.get()=='integer':
              md=int(E6T1.get())
           
         min_samples_leaf=int(E7T1.get())
         min_samples_split=int(E8T1.get())
         n_estimators=int(E9T1.get())
                  
         estimator=RandomForestRegressor(bootstrap=Criterion11.get(), criterion=Criterion9.get(),max_features=Criterion10.get(),
                                     max_depth=md,min_samples_leaf=int(E7T1.get()),
                                     min_samples_split=int(E8T1.get()),n_estimators=int(E9T1.get()))
         #clf5.fit(Xtr,ytr)
         #print(clf5.score(Xtr,ytr))
         
    elif var6.get():
         lr=float(E14T1.get())
         mss=float(E12T1.get())
         msl=float(E11T1.get())
         if Criterion26.get()=='none':
            md=None
         elif Criterion26.get()=='integer':
              md=int(E10T1.get())
         ss=float(E15T1.get())
         ne=int(E13T1.get())
         rn='GBR'
         estimator=GradientBoostingRegressor(loss=Criterion23.get(),learning_rate=lr,min_samples_split=mss, min_samples_leaf=msl,
                                        max_depth=md, max_features=Criterion24.get(), criterion=Criterion22.get(), subsample=ss,
                                         n_estimators=ne)   
    else:
        pass
    return estimator,rn           

   

    
form = tk.Tk()
form.title("QSAR-Co-X (Module-3)")
form.geometry("700x650")

tab_parent = ttk.Notebook(form)

tab1 = ttk.Frame(tab_parent)


tab_parent.add(tab1,text="User specific non-linear model")

###Tab1#####
    
####TAB2#####
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

thirdLabelTabThreer2c1=Label(tab1, text='Correlation cut-off',font=("Helvetica", 12))
thirdLabelTabThreer2c1.place(x=50,y=70)
thirdEntryTabThreer3c1=Entry(tab1, width=7)
thirdEntryTabThreer3c1.place(x=180,y=70)


fourthLabelTabThreer4c1=Label(tab1, text='Variance cut-off',font=("Helvetica", 12))
fourthLabelTabThreer4c1.place(x=250,y=70)
fourthEntryTabThreer5c1=Entry(tab1, width=7)
fourthEntryTabThreer5c1.place(x=370,y=70)

forthLabelTabOne=tk.Label(tab1, text="CV(model predictability)",font=("Helvetica", 12))
forthLabelTabOne.place(x=450,y=70)
forthEntryTabOne = tk.Entry(tab1, width=7)
forthEntryTabOne.place(x=630,y=70)

var1= IntVar()
CB1=Checkbutton(tab1, text = "KNN",  variable=var1, \
                 onvalue = 1, offvalue = 0, font=("Helvetica", 12))
CB1.place(x=60,y=100)

L1T1= tk.Label(tab1, text="n_neighbors",font=("Helvetica", 10))
L1T1.place(x=25,y=130)
E1T1 = tk.Entry(tab1, width=10)
E1T1.place(x=103,y=133)

Criterion2 = StringVar()
Criterion3 = StringVar()

L2T1= tk.Label(tab1, text="Weights",font=("Helvetica", 10))
L2T1.place(x=25,y=160)
C1= ttk.Radiobutton(tab1, text='Uniform', variable=Criterion2, value='uniform')
C2= ttk.Radiobutton(tab1, text='Distance', variable=Criterion2,  value='distance')
C1.place(x=80,y=160)
C2.place(x=160,y=160)

L3T1= tk.Label(tab1, text="Algorithm",font=("Helvetica", 10))
L3T1.place(x=20,y=190)
C3= ttk.Radiobutton(tab1, text='Auto', variable=Criterion3, value='auto')
C4= ttk.Radiobutton(tab1, text='Ball_tree', variable=Criterion3, value='ball_tree')
C5= ttk.Radiobutton(tab1, text='KD_tree', variable=Criterion3, value='kd_tree')
C6= ttk.Radiobutton(tab1, text='Brute', variable=Criterion3, value='brute')

C3.place(x=80,y=190)
C4.place(x=130,y=190)
C5.place(x=80,y=220)
C6.place(x=150,y=220)

b5=Button(tab1, text='Generate', command=sol,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b5.place(x=100,y=250)

#############################



########################################

var3= IntVar()
CB2=Checkbutton(tab1, text = "Support Vector Classification",  variable=var3, \
                 onvalue = 1, offvalue = 0, font=("Helvetica", 12))
CB2.place(x=200,y=100)

L6T1= tk.Label(tab1, text="C",font=("Helvetica", 10))
L6T1.place(x=270,y=130)
E3T1 = tk.Entry(tab1, width=10)
E3T1.place(x=303,y=133)

L7T1= tk.Label(tab1, text="Gamma",font=("Helvetica", 10))
L7T1.place(x=250,y=160)
E4T1 = tk.Entry(tab1, width=10)
E4T1.place(x=303,y=160)

Criterion5 = StringVar()

L8T1= tk.Label(tab1, text="Kernel",font=("Helvetica", 10))
L8T1.place(x=230,y=190)
C9= ttk.Radiobutton(tab1, text='Linear', variable=Criterion5, value='linear')
C10= ttk.Radiobutton(tab1, text='RBF', variable=Criterion5, value='rbf')
C9.place(x=290,y=190)
C10.place(x=370,y=190)

b6=Button(tab1, text='Generate', command=sol,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b6.place(x=295,y=210)

#################

var4= IntVar()
CB3=Checkbutton(tab1, text = "Multilayer Perception",  variable=var4, \
                 onvalue = 1, offvalue = 0, font=("Helvetica", 12))
CB3.place(x=30,y=300)

L9T1= tk.Label(tab1, text="alpha",font=("Helvetica", 10))
L9T1.place(x=60,y=330)
E5T1 = tk.Entry(tab1, width=10)
E5T1.place(x=100,y=333)

Criterion6 = StringVar()
Criterion7 = StringVar()
Criterion8 = StringVar()


L8T1= tk.Label(tab1, text="Activation",font=("Helvetica", 10))
L8T1.place(x=30,y=360)
C11= ttk.Radiobutton(tab1, text='Identity', variable=Criterion6, value='identity')
C12= ttk.Radiobutton(tab1, text='Logistic', variable=Criterion6, value='logistic')
C13= ttk.Radiobutton(tab1, text='Tanh', variable=Criterion6, value='tanh')
C14= ttk.Radiobutton(tab1, text='Relu', variable=Criterion6, value='relu')

C11.place(x=90,y=360)
C12.place(x=160,y=360)
C13.place(x=90,y=390)
C14.place(x=160,y=390)

L9T1= tk.Label(tab1, text="Solver",font=("Helvetica", 10))
L9T1.place(x=30,y=420)
C15= ttk.Radiobutton(tab1, text='SGD', variable=Criterion7, value='sgd')
C16= ttk.Radiobutton(tab1, text='Adam', variable=Criterion7, value='adam')
C15.place(x=90,y=420)
C16.place(x=160,y=420)

L10T1= tk.Label(tab1, text="Learning rate",font=("Helvetica", 10))
L10T1.place(x=10,y=450)
C17= ttk.Radiobutton(tab1, text='Constant', variable=Criterion8, value='constant')
C18= ttk.Radiobutton(tab1, text='Adaptive', variable=Criterion8, value='adaptive')
C19= ttk.Radiobutton(tab1, text='Invscaling', variable=Criterion8, value='invscaling')

C17.place(x=90,y=450)
C18.place(x=160,y=450)
C19.place(x=90,y=480)

b6=Button(tab1, text='Generate', command=sol,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b6.place(x=100,y=510)


###################
var5= IntVar()
CB3=Checkbutton(tab1, text = "Random Forest",  variable=var5, \
                 onvalue = 1, offvalue = 0, font=("Helvetica", 12))
CB3.place(x=280,y=270)

L11T1= tk.Label(tab1, text="Max depth",font=("Helvetica", 10))
L11T1.place(x=240,y=300)
Criterion25=StringVar()
Criterion25.set('none')
MaxFeatures_None = ttk.Radiobutton(tab1, text='None', variable=Criterion25, value='none')   
MaxFeatures_None.place(x=310,y=303)
MaxFeatures_integer = ttk.Radiobutton(tab1, text='Number', variable=Criterion25, value='integer')   
MaxFeatures_integer.place(x=370,y=303)                                          
E6T1 = tk.Entry(tab1,textvariable=MaxFeatures_integer, width=7)
E6T1.place(x=440,y=303)


L12T1= tk.Label(tab1, text="Minimum samples leaf",font=("Helvetica", 10))
L12T1.place(x=240,y=330)
E7T1 = tk.Entry(tab1, width=10)
E7T1.place(x=380,y=333)


L13T1= tk.Label(tab1, text="Minimum samples split",font=("Helvetica", 10))
L13T1.place(x=240,y=360)
E8T1 = tk.Entry(tab1, width=10)
E8T1.place(x=380,y=363)

L14T1= tk.Label(tab1, text="Number of Estimators",font=("Helvetica", 10))
L14T1.place(x=240,y=390)
E9T1 = tk.Entry(tab1, width=10)
E9T1.place(x=380,y=393)


Criterion9 = StringVar()
Criterion10 = StringVar()
Criterion11 = BooleanVar()


L15T1= tk.Label(tab1, text="Criterion",font=("Helvetica", 10))
L15T1.place(x=240,y=420)
C20= ttk.Radiobutton(tab1, text='MSE', variable=Criterion9, value='mse')
C21= ttk.Radiobutton(tab1, text='MAE', variable=Criterion9, value='mae')

C20.place(x=310,y=420)
C21.place(x=380,y=420)


L9T1= tk.Label(tab1, text="Max features",font=("Helvetica", 10))
L9T1.place(x=240,y=450)
C15= ttk.Radiobutton(tab1, text='Auto', variable=Criterion10, value='auto')
C16= ttk.Radiobutton(tab1, text='SQRT', variable=Criterion10, value='sqrt')
C16_1= ttk.Radiobutton(tab1, text='Log2', variable=Criterion10, value='log2')
C15.place(x=320,y=450)
C16.place(x=370,y=450)
C16_1.place(x=420,y=450)

L10T1= tk.Label(tab1, text="Bootstrap",font=("Helvetica", 10))
L10T1.place(x=240,y=480)
C17= ttk.Radiobutton(tab1, text='True', variable=Criterion11, value=True)
C18= ttk.Radiobutton(tab1, text='False', variable=Criterion11, value=False)

C17.place(x=330,y=480)
C18.place(x=380,y=480)


b6=Button(tab1, text='Generate', command=sol,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b6.place(x=330,y=510)

###################

var6= IntVar()
CB3=Checkbutton(tab1, text = "Gradient Boosting",  variable=var6, \
                 onvalue = 1, offvalue = 0, font=("Helvetica", 12))
CB3.place(x=500,y=140)

L11T1= tk.Label(tab1, text="Max depth",font=("Helvetica", 10))
L11T1.place(x=430,y=170)
Criterion26=StringVar()
Criterion26.set('none')
MaxFeatures1_None = ttk.Radiobutton(tab1, text='None', variable=Criterion26, value='none')   
MaxFeatures1_None.place(x=500,y=170)
MaxFeatures1_integer = ttk.Radiobutton(tab1, text='Number', variable=Criterion26, value='integer')   
MaxFeatures1_integer.place(x=560,y=170)                                          

E10T1 = tk.Entry(tab1, textvariable=MaxFeatures1_integer, width=7)
E10T1.place(x=630,y=170)

L12T1= tk.Label(tab1, text="Minimum samples leaf",font=("Helvetica", 10))
L12T1.place(x=470,y=200)
E11T1 = tk.Entry(tab1, width=10)
E11T1.place(x=610,y=203)


L13T1= tk.Label(tab1, text="Minimum samples split",font=("Helvetica", 10))
L13T1.place(x=470,y=230)
E12T1 = tk.Entry(tab1, width=10)
E12T1.place(x=610,y=233)

L14T1= tk.Label(tab1, text="Number of Estimators",font=("Helvetica", 10))
L14T1.place(x=470,y=260)
E13T1 = tk.Entry(tab1, width=10)
E13T1.place(x=610,y=263)

L14T1= tk.Label(tab1, text="Learning rate",font=("Helvetica", 10))
L14T1.place(x=520,y=290)
E14T1 = tk.Entry(tab1, width=10)
E14T1.place(x=610,y=290)

L14T1= tk.Label(tab1, text="Subsample",font=("Helvetica", 10))
L14T1.place(x=530,y=320)
E15T1 = tk.Entry(tab1, width=10)
E15T1.place(x=610,y=320)

Criterion22 = StringVar()
Criterion23 = StringVar()
Criterion24 = StringVar()


L15T1= tk.Label(tab1, text="Criterion",font=("Helvetica", 10))
L15T1.place(x=480,y=350)
C20= ttk.Radiobutton(tab1, text='Friedman MSE', variable=Criterion22, value='friedman_mse')
C21= ttk.Radiobutton(tab1, text='MAE', variable=Criterion9, value='mae')

C20.place(x=540,y=350)
C21.place(x=640,y=350)


L16T1= tk.Label(tab1, text="Loss",font=("Helvetica", 10))
L16T1.place(x=480,y=375)
C22= ttk.Radiobutton(tab1, text='Ls', variable=Criterion23, value='ls')
C23= ttk.Radiobutton(tab1, text='Lad', variable=Criterion23, value='lad')
C22_x= ttk.Radiobutton(tab1, text='Huber', variable=Criterion23, value='huber')
C23_x= ttk.Radiobutton(tab1, text='Quantile', variable=Criterion23, value='quantile')
C22.place(x=520,y=375)
C23.place(x=600,y=375)
C22_x.place(x=520,y=400)
C23_x.place(x=600,y=400)

L17T1= tk.Label(tab1, text="Max features",font=("Helvetica", 10))
L17T1.place(x=460,y=425)
C24= ttk.Radiobutton(tab1, text='Log2', variable=Criterion24, value='log2')
C25= ttk.Radiobutton(tab1, text='SQRT', variable=Criterion24, value='sqrt')
C26= ttk.Radiobutton(tab1, text='Auto', variable=Criterion24, value='auto')

C24.place(x=540,y=425)
C25.place(x=590,y=425)
C26.place(x=640,y=425)

b7=Button(tab1, text='Generate', command=sol,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b7.place(x=560,y=460)


tab_parent.pack(expand=1, fill='both')

form.mainloop()