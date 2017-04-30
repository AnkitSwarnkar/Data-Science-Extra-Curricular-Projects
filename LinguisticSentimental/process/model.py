# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier;
from sklearn import svm
#from 
import pandas as pd
import numpy as np

class Models:
        def init(self, model):
            if (model == "RandomForest"):
                self.model = RandomForestClassifier(n_estimators=100)
            elif(model=="Svm"):
                self.model = svm.SVC(kernel='linear', c=1, gamma=1)
            
        def getScore(self,xtrain, ytrain, xtest):
            print("Fitting the model.")
            self.model.fit(xtrain, ytrain)
            y = self.model.predict(xtest)
                
        
            
            
            
        
        
