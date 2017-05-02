import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
class bigramExtractor:
    def __init__(self):
        
        self.stop_words = set(stopwords.words("english"))
        self.vectorizer = 0
        
        self.vocab=[]
        
    def getBigram(self,tweets,test_train):    
        pos_list=[]

        i=0
        tweet_final=list()
        flatten_list=[]
        concat_list=[]
        def find_ngrams(input_list, n):
            return list(zip(*[input_list[i:] for i in range(n)]))
        for line in tweets:
            line_clean=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",line).split())
            line_clean=re.sub(r'\d+', '',line_clean)
            word_list=line_clean.split()
            #Remove stop words
            word_list=[w for w in word_list if not w in stopwords.words('english')]
            p = find_ngrams(word_list,2)
            #print(p)
            pos_list.append(p)
            line_concat_list = list(('_'.join(item) for item in p))
            
            string1 = " "
            for i in line_concat_list:
                string1 = string1 + i + " "
            tweet_final.append(string1)
        #print(tweet_final)
        flatten_list=[item for sublist in pos_list for item in sublist]
        concat_list = list(('_'.join(item) for item in flatten_list))
        counter=Counter(concat_list)
        common_tuples=(counter.most_common(1665))
        for item in common_tuples:
            self.vocab.append(item[0])
       
        data=0
        if (test_train==0):
                #For train
            print(np.shape(tweets))
            self.vectorizer = TfidfVectorizer(vocabulary=self.vocab)
            data = self.vectorizer.fit_transform(tweet_final).toarray()
        else:
            if(self.vectorizer == 0):
                print("Error....")
                exit(-1)
            data = self.vectorizer.transform(tweet_final).toarray()
        return data
if __name__=="__main__":
    wrd = dict()
    wrd[0] = "Hi There life is so awesome"
    wrd[1] = "Papa God, i pray that You shower me with more patience.  #worththewait #SemST"
    wrd[2] = "Everyone believe in whatever they want. #Freedom #SemST"
    name = pd.DataFrame.from_dict(wrd, orient='index')
    #print(name[0])
    t1 = bigramExtractor()
    data=t1.getBigram(name[0])
    print(data)