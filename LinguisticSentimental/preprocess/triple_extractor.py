import textacy as tc
import pandas as pd
from spacy.en import English
from sklearn.feature_extraction.text import TfidfVectorizer
import types
class TripletExtractor:
    def __init__(self):
        self.nlp=English()
        self.vocab=list()
        self.vectorizer=0
        
    def getSVO(self,tweets,train_test):
        #Build Vocabulary
        for sente in tweets : 
            print(sente)
            word_model = self.nlp(sente)
            SVobject = tc.extract.subject_verb_object_triples(word_model)
            for tag in SVobject:
                if type(tag) is tuple:
                    for word in tag:
                        self.vocab.append(word)
            self.vocab=list(set(self.vocab)) #remove duplicate
            #print (np.shape(pos_list_uniq))
        data=0
        if (train_test==0):
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
    t1 = TripletExtractor()
    t1.getSVO(name[0])
        
