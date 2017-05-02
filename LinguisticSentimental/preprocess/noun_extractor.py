# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 14:22:45 2017

@author: Ajinkya
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.en import English

class NounExtract:
    def __init__(self):
        self.nlp = English()
        self.vectorizer =0
        pass
        
    
    def nounExtactI(self,tweets, test_train):
        pos_list=[]
        train_test=test_train
        for line in tweets:
            line_clean=' '.join(re.sub('[^A-Za-z]+',' ',line).split())
        doc = self.nlp(line_clean)
        for word in doc:
            if word.pos_=="NOUN" or word.pos_=="ADJ" or word.pos_=="VERB" :
                pos_list.append(word.text)

        #print(pos_list)
        pos_list_uniq=list(set(pos_list)) #remove duplicate
        #print (np.shape(pos_list_uniq))
        data=0
        if (train_test==0):
            #For train
            self.vectorizer = TfidfVectorizer(vocabulary=pos_list_uniq)
            data = self.vectorizer.fit_transform(tweets).toarray()
        else:
            if(self.vectorizer == 0):
                print("Error....")
                exit(-1)
            data = self.vectorizer.transform(tweets).toarray()
        return data