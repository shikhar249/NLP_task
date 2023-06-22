import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

def bow(vocab, vectors, terms):
 df2 = pd.DataFrame(columns = ['term1', 'term2'])

 for i in range(len(terms)):
  if terms['term1'].iloc[i] in vocab and terms['term2'].iloc[i] in vocab:
   new_row = {'term1': terms['term1'].iloc[i], 'term2': terms['term2'].iloc[i]}
   df2 = pd.concat([df2, pd.DataFrame([new_row])], ignore_index = True)

 t1, t2 = [], []

 for i in range(len(df2)):
  t1_ind = np.where(vocab == df2['term1'].iloc[i])
  t1.append(vectors[:, t1_ind])
  t2_ind = np.where(vocab == df2['term2'].iloc[i])
  t2.append(vectors[:, t2_ind])

 t1 = np.squeeze(t1)
 t2 = np.squeeze(t2)
 
 cos_bow = np.diag(cosine_similarity(t1, t2)).mean()

 return cos_bow

def tfidf(vocab, vectors, terms):

 tfdf2 = pd.DataFrame(columns = ['term1', 'term2'])

 for i in range(len(terms)):
  if terms['term1'].iloc[i] in vocab and terms['term2'].iloc[i] in vocab:
   new_row = {'term1': terms['term1'].iloc[i], 'term2': terms['term2'].iloc[i]}
   tfdf2 = pd.concat([tfdf2, pd.DataFrame([new_row])], ignore_index = True)

 tft1, tft2 = [], []
 
 for i in range(len(tfdf2)): 
  tft1_ind = np.where(vocab == tfdf2['term1'].iloc[i])
  tft1.append(vectors[:, tft1_ind])
  tft2_ind = np.where(vocab == tfdf2['term2'].iloc[i])
  tft2.append(vectors[:, tft2_ind])

 tft1 = np.squeeze(tft1)
 tft2 = np.squeeze(tft2)

 cos_tf = np.diag(cosine_similarity(tft1, tft2)).mean()

 return cos_tf

def w2v(vocab, vectors, terms):

 twdf2 = pd.DataFrame(columns = ['term1', 'term2'])

 for i in range(len(terms)):
  if terms['term1'].iloc[i] in vocab and terms['term2'].iloc[i] in vocab:
   new_row = {'term1': terms['term1'].iloc[i], 'term2': terms['term2'].iloc[i]}
   twdf2 = pd.concat([twdf2, pd.DataFrame([new_row])], ignore_index = True)

 tw1, tw2 = [], []

 for i in range(len(twdf2)):
  tw1_ind = vocab.index(twdf2['term1'].iloc[i])
  tw1.append(vectors[tw1_ind, :])
  tw2_ind = vocab.index(twdf2['term2'].iloc[i])
  tw2.append(vectors[tw2_ind, :])

 tw1 = np.array(tw1)
 tw2 = np.array(tw2)

 cos_wv = np.diag(cosine_similarity(tw1, tw2)).mean()

 return cos_wv

def bert(vocab, vectors, terms):

 tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

 term1 = list(terms['term1'])
 term2 = list(terms['term2'])

 term1_tok = tokenizer(term1)
 term2_tok = tokenizer(term2)

 tk1, tk2 = [], []

 for i in range(len(term1_tok['input_ids'])):
  tk1.append(tokenizer.convert_ids_to_tokens(term1_tok['input_ids'][i]))

 for i in range(len(term2_tok['input_ids'])):
  tk2.append(tokenizer.convert_ids_to_tokens(term2_tok['input_ids'][i]))

 vec1, vec2 = [], []

 for ls1, ls2 in zip(tk1, tk2):
  ind1, ind2 = [], []
  for i in range(1, len(ls1)-1):
   try:
    ind1.append(vocab.index(ls1[i]))
   except:
    pass
  for i in range(1, len(ls2)-1):
   try:
    ind2.append(vocab.index(ls2[i]))
   except:
    pass
  if len(ind1) == len(ls1)-2 and len(ind2) == len(ls2)-2:
    subvec1, subvec2 = [], []
    for i in ind1:
     subvec1.append(vectors[i].tolist())
    subvec1 = list(np.array(subvec1).mean(axis=0))
    for i in ind2:
     subvec2.append(vectors[i].tolist())
    subvec2 = list(np.array(subvec2).mean(axis=0))
    vec1.append(subvec1)
    vec2.append(subvec2)

 vec1_arr = np.array(vec1)
 vec2_arr = np.array(vec2)

 cos_bert = np.diag(cosine_similarity(vec1_arr, vec2_arr)).mean()

 return cos_bert
