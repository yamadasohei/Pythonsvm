
import sklearn
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import joblib
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(0)

#学習

def learningmain(): 
    # 1. reading data
    file = np.loadtxt('trainingdata4_05s_allb_4000.csv', delimiter=',') #学習用データ
    print(file)
    xtrain = file[:, :file.shape[1]-1]
    ttrain = file[:,file.shape[1]-1]
    print(xtrain)
    print(ttrain)

    # 2. learning, cross-validation

    predictor = SVC()
    predictor.fit(xtrain,ttrain) 
    joblib.dump(predictor,"predictor_svc.pkl",compress=True) 
     
    # 3. evaluating the performance of the predictor
    liprediction=predictor.predict(xtrain)
    table=sklearn.metrics.confusion_matrix(ttrain,liprediction)
    tn,fp,fn,tp=table[0][0],table[0][1],table[1][0],table[1][1]
    print("Training")
    print("TPR\t{0:.3f}".format(tp/(tp+fn)))
    print("SPC\t{0:.3f}".format(tn/(tn+fp)))
    print("PPV\t{0:.3f}".format(tp/(tp+fp)))
    print("ACC\t{0:.3f}".format((tp+tn)/(tp+fp+fn+tn)))
    print("MCC\t{0:.3f}".format((tp*tn-fp*fn)/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**(1/2)))
    print("F1\t{0:.3f}".format((2*tp)/(2*tp+fp+fn)))
    
    #table = pd.DataFrame(data=table, index=["shaver", "dryer", "electricbrush", "handycleaner", "drilldriver", "handyfan", "handymassager"], columns=["shaver", "dryer", "electricbrush", "handycleaner", "drilldriver", "handyfan", "handymassager"])
    #table = pd.DataFrame(data=table, index=["shaver", "dryer", "electricbrush", "handycleaner", "drilldriver", "handyfan"], columns=["shaver", "dryer", "electricbrush", "handycleaner", "drilldriver", "handyfan"])
    #table = pd.DataFrame(data=table, index=["shaver", "dryer", "electricbrush", "handycleaner", "drilldriver"], columns=["shaver", "dryer", "electricbrush", "handycleaner", "drilldriver"])
    #table = pd.DataFrame(data=table, index=["super_shaver", "radicon", "shredder", "mixer", "drilldriver", "ultrabrush"], columns=["super_shaver", "radicon", "shredder", "mixer", "drilldriver", "ultrabrush"])
    table = pd.DataFrame(data=table, index=["super_shaver", "shredder", "mixer", "drilldriver", "ultrabrush"], columns=["super_shaver", "shredder", "mixer", "drilldriver", "ultrabrush"])
    plt.figure(figsize=(13, 10))
    sns.heatmap(table, square=True, cbar=True, annot=True, cmap='Blues')
    plt.yticks(rotation=0)
    plt.xlabel("Predicted", fontsize=18)
    plt.ylabel("Actual", fontsize=18)
    plt.savefig("confusion_matrix_trainingdata4_05s_allb_4000.pdf") #混同行列画像

    
    # 4. printing parameters of the predictor

    print(sorted(predictor.get_params(True).items()))
    print(predictor.support_vectors_)
    sv = pd.DataFrame(predictor.support_vectors_)
    sv.to_csv('support_vectors_.csv')
    print(predictor.dual_coef_)
    dc = pd.DataFrame(predictor.dual_coef_)
    dc.to_csv('dual_coef_.csv')
    print(predictor.intercept_)
    pi = pd.DataFrame(predictor.intercept_)
    pi.to_csv('intercept_.csv')
    print(predictor.gamma)
 
def classmain(): 
    # 1. reading data
    file = np.loadtxt('testdata4_05s_allb_4000.csv', delimiter=',') #テストデータ
    print(file)
    xtest = file[:, :file.shape[1]-1]
    ttest = file[:,file.shape[1]-1]
    print(xtest)
    print(ttest)


    # 2. learning, cross-validation
    predictor=joblib.load("predictor_svc.pkl")
     
    # 3. evaluating the performance of the predictor
    liprediction=predictor.predict(xtest)
    table=sklearn.metrics.confusion_matrix(ttest,liprediction)
    tn,fp,fn,tp=table[0][0],table[0][1],table[1][0],table[1][1]
    print("Test")
    print("TPR\t{0:.3f}".format(tp/(tp+fn)))
    print("SPC\t{0:.3f}".format(tn/(tn+fp)))
    print("PPV\t{0:.3f}".format(tp/(tp+fp)))
    print("ACC\t{0:.3f}".format((tp+tn)/(tp+fp+fn+tn)))
    print("MCC\t{0:.3f}".format((tp*tn-fp*fn)/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**(1/2)))
    print("F1\t{0:.3f}".format((2*tp)/(2*tp+fp+fn)))
    
    #table = pd.DataFrame(data=table, index=["shaver", "dryer", "electricbrush", "handycleaner", "drilldriver", "handyfan", "handymassager"], columns=["shaver", "dryer", "electricbrush", "handycleaner", "drilldriver", "handyfan", "handymassager"])
    #table = pd.DataFrame(data=table, index=["shaver", "dryer", "electricbrush", "handycleaner", "drilldriver", "handyfan"], columns=["shaver", "dryer", "electricbrush", "handycleaner", "drilldriver", "handyfan"])
    #table = pd.DataFrame(data=table, index=["shaver", "dryer", "electricbrush", "handycleaner", "drilldriver"], columns=["shaver", "dryer", "electricbrush", "handycleaner", "drilldriver"])
    #table = pd.DataFrame(data=table, index=["super_shaver", "radicon", "shredder", "mixer", "drilldriver", "ultrabrush"], columns=["super_shaver", "radicon", "shredder", "mixer", "drilldriver", "ultrabrush"])
    table = pd.DataFrame(data=table, index=["super_shaver", "shredder", "mixer", "drilldriver", "ultrabrush"], columns=["super_shaver", "shredder", "mixer", "drilldriver", "ultrabrush"])
    plt.figure(figsize=(13, 10))
    sns.heatmap(table, square=True, cbar=True, annot=True, cmap='Blues')
    plt.yticks(rotation=0)
    plt.xlabel("Predicted", fontsize=18)
    plt.ylabel("Actual", fontsize=18)
    plt.savefig("confusion_matrix_testdata4_05s_allb_4000.pdf")

    
if __name__ == '__main__':
    learningmain()
    classmain()