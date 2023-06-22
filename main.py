import pandas as pd
import preprocessing
import vectorization
import similarity

clinnotes = pd.read_csv('ClinNotes.csv')
concepts = pd.read_csv('MedicalConcepts.csv')

doc, term = preprocessing.text_clean(clinnotes, concepts)


vocab_bow, vectors_bow = vectorization.bow(doc['notes'])

vocab_tfidf, vectors_tfidf = vectorization.tfidf(doc['notes'])

vocab_wv, vectors_wv = vectorization.w2v(doc['notes'])

vocab_bert, vectors_bert = vectorization.bert(doc['notes'])


print('Bag of Words: ', similarity.bow(vocab_bow, vectors_bow, term))
print('TF-IDF: ', similarity.tfidf(vocab_tfidf, vectors_tfidf, term))
print('Word2Vec: ', similarity.w2v(vocab_wv, vectors_wv, term))
print('BERT: ', similarity.bert(vocab_bert, vectors_bert, term))


