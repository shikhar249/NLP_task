import spacy
import en_core_sci_sm
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

def text_clean(clinnotes, concepts):
 clinnotes['notes'] = clinnotes['notes'].str.lower()
 clinnotes['notes'] = clinnotes['notes'].str.replace(',',' , ')
 clinnotes['notes'] = clinnotes['notes'].str.replace('-year-old', ' year old')
 diction = {',':' , ', '-year-old':' year old', ' lv ':' left ventricular ', ' yo ':' year old ', ' pa ': ' pulmonary artery '}
 for x in diction:
     clinnotes['notes'] = clinnotes['notes'].str.replace(x, diction[x])
 clinnotes['notes'] = clinnotes['notes'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))

 concepts['Term1'] = concepts['Term1'].str.lower()
 concepts['Term2'] = concepts['Term2'].str.lower()
 concepts.columns = concepts.columns.str.lower()

 # removing stop words
 stopwords_list = set(stopwords.words('english'))
 clinnotes['notes'] = clinnotes['notes'].apply(lambda x: word_tokenize(x))
 print(clinnotes['notes'])
 clinnotes['notes'] = clinnotes['notes'].apply(lambda x: [word for word in x if word not in stopwords_list])
 print(clinnotes['notes'])
 clinnotes['notes'] = clinnotes['notes'].apply(lambda x: [word if not word.isdigit() else 'NUM' for word in x])

 clinnotes['notes'] = clinnotes['notes'].apply(lambda x: ' '.join(x))

 nlp_sm = spacy.load('en_core_sci_sm')

 for i in range(len(clinnotes)):
  clinnotes['notes'].iloc[i] = str(nlp_sm(clinnotes['notes'].iloc[i]))
  
   for i in range(len(concepts)):
  concepts['term1'].iloc[i] = str(nlp_sm(concepts['term1'].iloc[i]))
  concepts['term2'].iloc[i] = str(nlp_sm(concepts['term2'].iloc[i]))

 return clinnotes, concepts
