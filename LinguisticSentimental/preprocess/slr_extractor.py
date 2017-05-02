import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
class SRLextractor:
    def __init__(self):
        self.vectorizer = 0
        self.vocab = []
        self.english_vocab = set(w.lower() for w in nltk.corpus.words.words())
        self.stopword = stopwords.words('english')
        
    def getSLRfeature(self,tweets,test_train):
        objlist = pd.read_excel("/home/ankit/Documents/Ling/LinguisticSentimental/preprocess/SLROBJ.xlsx")
        objlist.dropna(inplace = True)
        #Create Vocab 
        for line in  objlist['SLR']:
            for word in line.split():
                if word in self.english_vocab and len(word) > 1 and word not in self.stopword:
                    self.vocab.append(word)
            
        #print(self.vocab)
        self.vocab=list(set(self.vocab)) #remove duplicate
        #print(np.shape(self.vocab))
        #print(self.vocab)
        #print (np.shape(pos_list_uniq))
        data=0
        if (test_train==0):
                #For train
            self.vectorizer = TfidfVectorizer(vocabulary=self.vocab)
            data = self.vectorizer.fit_transform(tweets).toarray()
        else:
            if(self.vectorizer == 0):
                print("Error....")
                exit(-1)
            data = self.vectorizer.transform(tweets).toarray()
        return data
        
if __name__=="__main__":
    wrd = dict()
    wrd[0] = "Hi There life is so awesome"
    wrd[1] = "Papa God, i pray that You shower me with more patience.  #worththewait #SemST"
    wrd[2] = "Everyone believe in whatever they want. #Freedom #SemST"
    name = pd.DataFrame.from_dict(wrd, orient='index')
    #print(name[0])
    t1 = SRLextractor()
    data=t1.getSLRfeature(name[0],0)
    print(data)