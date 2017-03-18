import pandas as pd
import sys
import os
import re, string
from string import digits
class DataLoader:

    def __init__(self,location):
        self.location=location
        pass
    def clean_punc(self,sentence):
        #Remove number 
        remove_digits = str.maketrans('', '', digits)
        sentence = sentence.translate(remove_digits)
        #Remove Extra enter 
        sentence = sentence.lower().replace('\n', ' ').replace('\r', ' ')
        sentence = sentence.replace('  ',' ')
        #Remove punctuation
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        return regex.sub('', sentence)
        
    def load_data(self,dtype):
        locationval = self.location + "\\"+ dtype
        if os.path.isdir(locationval) is False:
            print("[ERROR#403] File not found")
            sys.exit(-1)
        data = dict()
        for file in os.listdir(locationval):
            with open(locationval + "\\" + file) as f:
                test = f.read()
                test = self.clean_punc(test)
                data[file] = test
        #Transfer dictionary to dataframe
        data_pd = pd.DataFrame.from_dict(data,orient='index')
        data_pd.columns = ['review']
        data_pd['label'] = dtype
        return data_pd
        
            
        
        