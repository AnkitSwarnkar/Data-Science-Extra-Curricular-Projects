import pandas as pd
import numpy as np
from preprocess.dataloader import loader
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import svm

if __name__=="__main__":
    h1 = loader("data/")
    X_train_tfidf, Y_train, X_test_tfidf, test_y = h1.getdata()
    #clf = MultinomialNB().fit(X_train_tfidf, Y_train)
    #clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42).fit(X_train_tfidf, Y_train)
    clf = svm.LinearSVC().fit(X_train_tfidf, Y_train)
    
    prediction = clf.predict(X_test_tfidf)
    #print(prediction)
    #print(test_y)
    y = np.bincount(prediction)
    ii = np.nonzero(y)[0]
    for i in zip(ii,y[ii]) :
        print(i)
    a1=(np.sum(prediction==test_y))/len(prediction)
    print(a1)
    
    
    
