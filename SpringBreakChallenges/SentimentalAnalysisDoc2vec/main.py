import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from DataLoader import DataLoader 
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from sklearn.linear_model import SGDClassifier

# to have a label associated
def labelGensim(data, label):
    final_label=[]
    for i,data in enumerate(data):
        label = '%s_%s'%(label,i)
        final_label.append(LabeledSentence(data, [label]))
    return final_label  

if __name__=="__main__" :
     #Create Dataset
     dataloader = DataLoader("data")
     negative_pd = dataloader.load_data("neg")
     positive_pd = dataloader.load_data("pos")
     #Now we have two dataframe for negative and positive
     dataset=pd.concat([positive_pd,negative_pd])
     df = dataset.sample(frac=1).reset_index(drop=True)
     #Lets divide the data in test and train
     X_train, X_test, y_train, y_test = train_test_split(df['review'],
                                                         df['label'])
     X_train = labelGensim(X_train, 'TRAIN')
     X_test = labelGensim(X_test, 'TEST')
     
     #Initiate our model 
     #size: Dimention of feature
     #min_count is  ignore all words with total frequency lower than this| 
     #Window: the maximum distance between the current and predicted word within a sentence
     #sample: threshold for configuring which higher-frequency words are randomly downsampled
     #workers: use this many worker threads to train the model
     model_dm = Doc2Vec(min_count=1, 
                                        window=10, size=100, sample=1e-3, 
                                        negative=5, workers=3)
     model_dbow = Doc2Vec(min_count=1, window=10, size=100, sample=1e-3, 
                                       negative=5, dm=0, workers=3)
     
     
     # build the vocabulary table
     vocab = np.concatenate((X_train, X_test))
     model_dm.build_vocab(vocab)
     model_dbow.build_vocab(vocab)
     
     #Train the Doc2vec
     for epoch in range(10):
         perm = np.random.permutation(X_train.shape[0])
         model_dm.train(X_train[perm])
         model_dbow.train(X_train[perm])
        
     #model_dm.save('model_dm.d2v')
     #model_dbow.save('model_dbow.d2v')
     #Get Vector
     vecs_dm = [np.array(model_dm[z.labels[0]]).reshape((1, 100)) 
     for z in X_test]
     vecs_dbow = [np.array(model_dm[z.labels[0]]).reshape((1, 100)) 
     for z in X_test]
     vec_test = np.hstack((vecs_dm,vecs_dbow))
     
     ##TRAIN AND TEST the model using SGDC
     lm = SGDClassifier(loss='log', penalty='l1')
     lm.fit(train_vecs, y_train)
     
     