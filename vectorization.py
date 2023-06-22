import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel

def bow(text):
 
 vectorizer = CountVectorizer()
 vectors = vectorizer.fit_transform(text)

 dense_vectors = vectors.todense()

 return vectorizer.get_feature_names_out(), dense_vectors

def tfidf(text):

 tf_vectorizer = TfidfVectorizer()
 tf_vector = tf_vectorizer.fit_transform(text)

 dense_tf_vectors = tf_vector.todense()

 return tf_vectorizer.get_feature_names_out(), dense_tf_vectors

def w2v(text):
 
 ls = []

 for i in range(len(text)):
  ls.append(word_tokenize(text.iloc[i]))

 model = Word2Vec(ls, min_count=1)
 vocab = model.wv.index_to_key
 dense_tw_vectors = model.wv.vectors

 return vocab, dense_tw_vectors

def bert(text):
 
 tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
 model = AutoModel.from_pretrained('bert-base-uncased')

 max_length = 128

 tokens = []

 for i in range(len(text)):
  tokens.append(tokenizer.tokenize(text.iloc[i]))

 for i in range(len(tokens)):
  tokens[i] = ['[CLS]'] + tokens[i][:max_length - 2] + ['[SEP]']
  tokens[i]  = ' '.join(tokens[i])

 inputs = tokenizer(tokens, padding=True, truncation=True, return_tensors='pt')

 batch_size = 16

 input_ids_batches = torch.split(inputs['input_ids'], batch_size)
 attention_mask_batches = torch.split(inputs['attention_mask'], batch_size)

 embeddings = []
 sen_embed = []
 for input_ids_batch, attention_mask_batch in zip(input_ids_batches, attention_mask_batches):
    with torch.no_grad():
        embeddings.append(model(input_ids=input_ids_batch, attention_mask=attention_mask_batch).last_hidden_state)
        sen_embed.append(model(input_ids=input_ids_batch,  attention_mask=attention_mask_batch).pooler_output)
 embeddings = list(torch.cat(embeddings, dim=0))

 words = []

 for input_ids in inputs['input_ids']:
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    words.append(tokens)

 flat_words = [item for sublist in words for item in sublist]
 flat_emb = [item for sublist in embeddings for item in sublist]

 return flat_words, flat_emb
