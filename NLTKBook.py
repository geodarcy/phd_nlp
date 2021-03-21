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

train_clean_sentences = []
for key in all_text_docs.keys():
  cleaned = clean(all_text_docs[key])
  cleaned = ' '.join(cleaned)
  train_clean_sentences.append(cleaned)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=0.1)
X = vectorizer.fit_transform(train_clean_sentences)
modelkmeans = KMeans(init='k-means++', max_iter=200, n_init=100)
modelkmeans.fit(X)
order_centroids = modelkmeans.cluster_centers_.argsort()[:,::-1]
terms = vectorizer.get_feature_names()
for i in range(modelkmeans.n_clusters):
  print("Cluster {}:".format(i)),
  for ind in order_centroids[i,:10]:
    print("{}".format(terms[ind]))

s = all_text_docs[name]
tokens = word_tokenize(s)
text = nltk.Text(tokens)
text.collocations()
text.concordance('social')
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(word_tokenize(s))
finder.nbest(bigram_measures.pmi, 10)


######
NUM_TOPICS = 10

vectorizer = CountVectorizer(min_df=5, max_df=0.9,
                             stop_words='english', lowercase=True)
data_vectorized = vectorizer.fit_transform(train_clean_sentences)

# Build a Latent Dirichlet Allocation Model
lda_model = LatentDirichletAllocation(n_topics=NUM_TOPICS, max_iter=10, learning_method='online')
lda_Z = lda_model.fit_transform(data_vectorized)

text = "Carbon pricing affects mobility and social exclusion"
x = lda_model.transform(vectorizer.transform([text]))[0]
print(x, x.sum())
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda_model, data_vectorized, vectorizer, mds='tsne')
pyLDAvis.show(panel)

## calculate term frequencies
docDF = DataFrame.from_dict(all_text_docs, orient='index', columns=['orig_text'])
docDF['clean_text'] = docDF['orig_text'].apply(lambda x: clean(x))
docDF['clean_len'] = docDF['clean_text'].apply(lambda x: len(x))
terms = ['carbon', 'pricing', 'greenhouse', 'backstop', 'infrastructure', 'fuel', 'gas', 'transportation', 'transit', 'mobility', 'pass', 'social', 'exclude', 'alone', 'friend', 'family']
for term in terms:
  docDF[term] = docDF['clean_text'].apply(lambda x: x.count(term))
docDF[terms].hist()
plt.show()

## guided LDA
vectorizer = CountVectorizer(min_df=5, max_df=0.9,
                             stop_words='english', lowercase=True, ngram_range=(1,3))
data_vectorized = vectorizer.fit_transform(train_clean_sentences)
all_text = ' '.join(train_clean_sentences)
tokens = word_tokenize(all_text)
words = [w.lower() for w in tokens]
vocab = sorted(set(words))
word2id = dict((v, idx) for idx, v in enumerate(vocab))
seed_topic_list = [['carbon', 'pricing', 'greenhouse', 'backstop', 'infrastructure'],
                   ['mobility', 'transit', 'transportation'],
                   ['social', 'exclusion', 'alone', 'friend', 'family']]
model = guidedlda.GuidedLDA(n_topics=4, n_iter=100, refresh=20)
seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
  for word in st:
    seed_topics[word2id[word]] = t_id

model.fit(data_vectorized, seed_topics=seed_topics, seed_confidence=0.15)
n_top_words = 10
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
  topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
  print('Topic {}: {}'.format(i, ' '.join(topic_words)))


## try Word2Vec
gensim_list = []
sizes = range(50,550,50)
termsDF = DataFrame(index=pd.MultiIndex.from_product([terms, sizes], names=['terms', 'sizes']), columns=["Term" + str(x) for x in range(1,11)])
for key in all_text_docs.keys():
  gensim_list.append(gensim.utils.simple_preprocess(all_text_docs[key]))
bigrams = gensim.models.phrases.Phrases(gensim_list)
for size in sizes:
  model = gensim.models.Word2Vec(bigrams[gensim_list], size=size, window=10, min_count=10, workers=10)
  model.train(gensim_list, total_examples=len(gensim_list), epochs=10)
  for term in terms:
    try:
      termsDF.loc[(term, size)] = [i for (i,v) in model.wv.most_similar(positive=term)]
    except:
      print("{} not in vocabulary".format(term))
termsDF.to_csv(os.path.join(path, 'Word2VecOutput', 'TermsDF.csv'))
model.wv.similarity(w1='gas', w2='car')

vocab = list(model.wv.vocab)
X = model[vocab]
pca = PCA(n_components=3, random_state=11, whiten=True)
tsne = TSNE(n_components=3, random_state=11)
clf = pca.fit_transform(X)

plot_terms = ['Term' + str(x) for x in range(1,4)]
terms_results = pd.unique(termsDF.loc[:,plot_terms].values.ravel('K'))
tmp = pd.DataFrame(clf, index=vocab, columns=['x', 'y', 'z'])
tmp = tmp.loc[set(terms + list(terms_results))]
fig = plt.figure(figsize=(11, 11))
ax = fig.add_subplot(111)

ax.scatter(tmp['x'], tmp['y'], alpha = 0.5)

for word, row in tmp.iterrows():
    x, y, z = row
    pos = (x, y)
    ax.text(x, y, s=word, size=8, zorder=1, color='k')

plt.title('w2v map - PCA')
# plt.savefig(os.path.join(path, 'Word2VecOutput', 'PCAPlot.png'))
plt.show()
