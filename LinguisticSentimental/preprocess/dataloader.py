# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import numpy as np
import nltk
import re
from preprocess.noun_extractor import NounExtract
from preprocess.triple_extractor import TripletExtractor
from preprocess.bigram_extractor import bigramExtractor
from preprocess.slr_extractor import SRLextractor
class loader:
    def __init__(self, working_dir=""):
        if (working_dir==""):
            self.working_dir = "../data/"
        else:
            self.working_dir = working_dir
        self.test_file  = self.working_dir + "testStance.xlsx"
        self.train_file = self.working_dir + "trainStance.xlsx"
        self.tokenizer = nltk.word_tokenize
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.noun_extract = NounExtract()
        self.triplet_extract = TripletExtractor()
        self.bigram_extract = bigramExtractor()
        self.srl_extract = SRLextractor()
        
    def clean(self, row):
        #Example = "dear lord thank u for all of ur blessings forg"
        dataread = row['Tweet']
        dataread = re.sub('[^A-Za-z]+',' ',dataread)
        dataread = str.lower(dataread)
        row['Tweet'] = dataread
    #For Target 
        if row['Target'] == "Legalization of Abortion" :
            row['Target'] = 1
        elif row['Target'] == "Hillary Clinton" :
            row['Target'] = 2
        elif row['Target'] == "Atheism" :
            row['Target'] = 3
        elif row['Target'] == "Feminist Movement" :
            row['Target'] = 4
        elif row['Target'] == "Climate Change" :
            row['Target'] = 5
        else:
            row['Target'] = 6
    
   #For Sentiment
        if (row['Sentiment'] == 'POSITIVE'):
            row['Sentiment']= 1
        elif row['Sentiment'] == 'NEGATIVE':
            row['Sentiment']= 2
        else:
            row['Sentiment'] = 0

    # For Opinion
        if row['Opinion towards'] == 'NO ONE':
            row['Opinion towards']= 1
        elif row['Opinion towards'] == 'TARGET':
            row['Opinion towards'] = 2
        else:
            row['Opinion towards'] = 0
        
        if row['Stance'] == 'FAVOR':
            row['Stance']= 1
        elif row['Stance'] == 'AGAINST':
            row['Stance']= 2
        else:
            row['Stance'] = 0
        return row
            
        
    def tdidfcreator(self,data,ngram_range_high=1,ngram_range_low=1):
        #print(np.shape(data))
        self.vectorizer = TfidfVectorizer(tokenizer=self.tokenizer, 
                                           stop_words=self.stopwords,
                                           min_df=1, 
                                           ngram_range=(ngram_range_high, 
                                                        ngram_range_low))
    
        res = self.vectorizer.fit_transform(data)
        #print(np.shape(res))
        return res.toarray()
    
    def tdidfcreatortest(self,data):
        #print(np.shape(data))
        res = self.vectorizer.transform(data)
        #print(np.shape(res))
        return res.toarray()
        
        
    def getdata(self):
        dataset_train = pd.read_excel(self.train_file)
        #print(np.shape(dataset_train))
        dataset_test = pd.read_excel(self.test_file)
        #Data Cleaning for train
        dataset_train = dataset_train.apply(self.clean,axis=1)
        tweetunigramfeature = self.tdidfcreator(dataset_train['Tweet'])
        
        for i in ['Target','Sentiment']:
            data = dataset_train[i].values.reshape((-1,1))
            #print(np.shape(data))
            tweetunigramfeature = np.concatenate((tweetunigramfeature,data)
            , axis =1)
        noun_feature = self.noun_extract.nounExtactI(dataset_train['Tweet'],
                                                     test_train=0)
        triplet_feature = self.triplet_extract.getSVO(dataset_train['Tweet'],
                                                      test_train=0)
        bigram_feature = self.bigram_extract.getBigram(dataset_train['Tweet'],
                                                      test_train=0)
        srl_feature = self.srl_extract.getSLRfeature(dataset_train['Tweet'],
                                                      test_train=0)
        #print(np.shape(triplet_feature))
        tweetunigramfeature = np.concatenate((tweetunigramfeature,noun_feature)
            , axis =1)
        tweetunigramfeature = np.concatenate((tweetunigramfeature,triplet_feature)
            , axis =1)
        tweetunigramfeature = np.concatenate((tweetunigramfeature,bigram_feature)
            , axis =1)
        tweetunigramfeature = np.concatenate((tweetunigramfeature,srl_feature)
            , axis =1)
        print("Returing Dimension : ",np.shape(tweetunigramfeature))
        #dataset_train = dataset_train.drop('Tweet', 1)
         # For test dataset_test
        dataset_test=dataset_test.apply(self.clean,axis = 1)
        testtweetunigramfeature = self.tdidfcreatortest(dataset_test['Tweet'])
        for i in ['Target','Sentiment']:
            data = dataset_test[i].values.reshape((-1,1))
            #print(np.shape(data))
            testtweetunigramfeature = np.concatenate((testtweetunigramfeature,data)
            , axis =1)
        noun_feature = self.noun_extract.nounExtactI(dataset_test['Tweet'],
                                                     test_train=1)
        triplet_feature = self.triplet_extract.getSVO(dataset_test['Tweet'],
                                                      test_train=1)
        bigram_feature = self.bigram_extract.getBigram(dataset_test['Tweet'],
                                                      test_train=1)
        srl_feature = self.srl_extract.getSLRfeature(dataset_test['Tweet'],
                                                      test_train=1)
        testtweetunigramfeature = np.concatenate((testtweetunigramfeature,
                                                  noun_feature)
            , axis =1)
        testtweetunigramfeature = np.concatenate((testtweetunigramfeature,
                                                  triplet_feature)
            , axis =1)
        testtweetunigramfeature = np.concatenate((testtweetunigramfeature,
                                                  bigram_feature)
            , axis =1)
        testtweetunigramfeature = np.concatenate((testtweetunigramfeature,
                                                  srl_feature)
            , axis =1)
        return tweetunigramfeature,dataset_train['Stance'],testtweetunigramfeature,dataset_test['Stance'] ;
        #First feature would be unigram. tf idf of tweet
if __name__ == "__main__":
    n1=loader()
    x1,x2,x3,x4 = n1.getdata()
    print(np.shape(x1))