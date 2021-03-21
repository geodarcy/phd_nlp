import os
import re
import string
import numpy as np
import pandas as pd
from pandas import DataFrame
from collections import Counter
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import nltk
from nltk.tokenize import regexp_tokenize, wordpunct_tokenize, blankline_tokenize, TreebankWordTokenizer
from nltk.stem import PorterStemmer # import Porter stemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
import guidedlda
from nltk.collocations import BigramCollocationFinder
from textblob import TextBlob
import gensim
import matplotlib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
matplotlib.style.use('ggplot')

path = '/Users/darcy/University/UofA/PhD/EAS/PhDWork/Studies/1 Policy/Documents'
wlem = WordNetLemmatizer()
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

# Cleaning the text sentences so that punctuation marks, stop words & digits are removed
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word.lower()) for word in punc_free.split())
    processed = re.sub(r"\d+","",normalized)
    x = processed.replace('\x00', '')
    y = x.split()
    return y

all_text_docs = {}
read_errors = []
for dirpath, dirnames, files in os.walk(os.path.join(path, 'ExtractedText')):
  for name in files:
    if name[-3:] == 'txt':
      try:
        with open(os.path.join(dirpath, name), 'r', encoding=encoding) as f:
          all_text_docs[name] = f.read()
          break
      except:
          read_errors.append(name)
all_docs = pd.read_pickle('/Users/darcy/University/UofA/PhD/EAS/PhDWork/Studies/1 Policy/Documents/AllDocsPickle')
all_docs['word_count'] = all_docs.doc_text.apply(lambda x: len(x.split()))
total_word_count = all_docs['word_count'].sum()
mobility_titles = pd.read_csv('/Users/darcy/University/UofA/PhD/EAS/PhDWork/Studies/1 Policy/Documents/LDAOutput/NPNMobilityTitles.csv')
social_titles = pd.read_csv('/Users/darcy/University/UofA/PhD/EAS/PhDWork/Studies/1 Policy/Documents/LDAOutput/NPNSocialTitles.csv')
mobility_list = mobility_titles['Unnamed: 0']
mobility_word_count = all_docs.loc[mobility_list, 'word_count'].sum()
social_list = social_titles['Unnamed: 0']
social_word_count = all_docs.loc[social_list, 'word_count'].sum()
print("Total words: {}".format(total_word_count))
print("Mobility words: {}".format(mobility_word_count))
print("Social words: {}".format(social_word_count))
