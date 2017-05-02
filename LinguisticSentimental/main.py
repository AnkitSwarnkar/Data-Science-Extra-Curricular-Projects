import pandas as pd
import numpy as np
from preprocess.dataloader import loader
from process.model import Models
from postprocess.report import Reporting

if __name__=="__main__":
    #Step 1 : get the data 
    h1 = loader("data/")
    X_train_tfidf, Y_train, X_test_tfidf, test_y = h1.getdata()
    
    #Select a model 
    #Selected SVM
    model = Models(model = "Svm")
    report = Reporting()
    #clf = MultinomialNB().fit(X_train_tfidf, Y_train)
    #clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42).fit(X_train_tfidf, Y_train)
    clf,prediction = model.getScore(X_train_tfidf, Y_train, X_test_tfidf, test_y)
    print("Overall : ",     clf)
    report.resultSummaryAnalysis(test_y,prediction)
    
    
    
    
