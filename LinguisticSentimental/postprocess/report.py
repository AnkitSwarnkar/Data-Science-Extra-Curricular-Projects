# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
class Reporting:
        def __init__(self,data= 0):
            self.train_data = pd.read_excel("data/trainStance.xlsx")
            self.test_data = pd.read_excel("data/testStance.xlsx")
            self.attributes = self.train_data['Target'].unique()
        
        def initial_report(self):
            traindata = self.train_data
            testdata = self.test_data
            
            #Count of each attribute
            print("Training Data Description Count")
            print(traindata.groupby('Target').count())
            print(traindata.groupby('Target').count()/len(traindata))
            print("Test Data Description Count")
            print(testdata.groupby('Target').count())
            print(testdata.groupby('Target').count()/len(testdata))
        
        def resultSummaryAnalysis(self, testdata, prediction):
            for val in self.attributes :
                print(val)
                k1 = np.where(self.test_data['Target']==val)[0]
                one = testdata[k1]
                #print(one)
                two = prediction[k1]
                #print(two)
                print(np.sum(one==two)/len(one))
                
            
            
            
        
if __name__=="__main__":
    h1=Reporting()
    #h1.initial_report()
    #h1.resultSummaryAnalysis()
    wrd = dict()
    wrd[0] = 1
    wrd[1] = 0
    wrd[2] = 2
    wrd[3] = 1
    wrd2 = dict()
    wrd2[0] = 1
    wrd2[1] = 0
    wrd2[2] = 2
    wrd2[3] = 2
       
    name1 = pd.DataFrame.from_dict(wrd, orient='index')
    name2 = pd.DataFrame.from_dict(wrd2, orient='index')
    #print(name[0])
    t1 = h1.resultSummaryAnalysis(name1,name2)
    #print(data)