import os
import re
import string
import numpy as np
from random import sample
import pandas as pd
from pandas import DataFrame
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

path = '/Users/darcy/University/UofA/PhD/EAS/PhDWork/Studies/1 Policy/Documents'
nltk.download('wordnet')
stemmer = SnowballStemmer('english')
french_words = ['et', 'de', 'la', 'ou', 'une', 'le', 'en', 'du', 'est', 'un', 'des', 'au', 'les', 'personne', 'qui', 'dans', 'pour', 'ce', 'sente', 'partie', 'sur', 'que', 'si', 'ministre', 'pe', 'il', 'peut', 'vertu', 'presente', 'paragraphe']
technical_words = ['mt', 'kt', 'kg', 'cm', 'http', 'www', 'nd', 'ht', 'dt', 'ta', 'ne']
stop = set(stopwords.words('english') + french_words + technical_words + [str(x) for x in range(3000)] + ['000'])
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
terms = ['carbon', 'pricing', 'climate', 'greenhouse', 'backstop', 'infrastructure', 'fuel', 'gas', 'transportation', 'transit', 'mobility', 'pass', 'social', 'exclude', 'alone', 'friend', 'family']
scales = ['Federal', 'Provincial', 'Municipal']
topic_count = 10

# Cleaning the text sentences so that punctuation marks, stop words & digits are removed
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word.lower()) for word in punc_free.split())
    processed = re.sub(r"\d+","",normalized)
    x = processed.replace('\x00', '')
    y = x.split()
    return y

all_text_docs = pd.read_pickle(os.path.join(path, 'AllDocsPickle'))
all_text_docs['clean_text'] = all_text_docs['doc_text'].apply(clean)
all_text_docs['Location'] = all_text_docs['path'].apply(lambda x: x.split('/')[-1])
all_text_docs['Scale'] = 'Provincial'
all_text_docs.loc[all_text_docs['Location'] == 'Canada', 'Scale'] = 'Federal'
all_text_docs.loc[all_text_docs['Location'] == 'Cities', 'Scale'] = 'Municipal'

# try Latent Dirichlet Allocation
# vect = CountVectorizer(max_features=100000, ngram_range=(1,2), stop_words=stop, max_df=0.4)
# for scale in scales:
#   text_train = list(all_text_docs.loc[all_text_docs['Scale'] == scale, 'doc_text'])
#   X = vect.fit_transform(text_train)
#   lda = LatentDirichletAllocation(n_topics=topic_count, learning_method="batch", max_iter=25, n_jobs=6)
#   document_topics = lda.fit_transform(X)
#   sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
#   feature_names = np.array(vect.get_feature_names())
#   # mglearn.tools.print_topics(topics=range(10), feature_names=feature_names, sorting=sorting, topics_per_chunk=5, n_words=10)
#   topicsDF = DataFrame(index=range(topic_count), columns=['Word_' + str(x) for x in range(10)])
#   for i in range(topic_count):
#     for j in range(10):
#       topicsDF.iloc[i,j] = vect.get_feature_names()[sorting[i][j]]
#   topicsDF.to_csv(os.path.join(path, 'LDAOutput', 'LDA' + scale + '.csv'))

# try again with gensim LDA
def lemmatize_stemming(text):
  return lemma.lemmatize(text, pos='v')
def preprocess(text):
  result = []
  for token in gensim.utils.simple_preprocess(text, deacc=True):
    if token not in stop and len(token) > 3:
      result.append(lemmatize_stemming(token))
  return result

scale_models = {} # need to save the model at each scale
all_text_docs['Gensim_processed'] = all_text_docs['doc_text'].map(preprocess)
all_processed_docs = all_text_docs['Gensim_processed']
dictionary = gensim.corpora.Dictionary(all_processed_docs)
dictionary.filter_extremes(no_below=2, no_above=0.3, keep_n=1000000)
# for scale in scales:
#   processed_docs = all_text_docs.loc[all_text_docs['Scale'] == scale, 'Gensim_processed']
#   bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
#   scale_models[scale] = gensim.models.LdaMulticore(bow_corpus, num_topics=topic_count, id2word=dictionary, passes=2, workers=6, random_state=10)
#   topics = scale_models[scale].show_topics(formatted=False)
#   topicsDF = DataFrame(index=range(topic_count), columns=['Word_' + str(x) for x in range(10)])
#   for i in range(topic_count):
#     for j in range(10):
#       topicsDF.iloc[i,j] = topics[i][1][j][0]
#   topicsDF.to_csv(os.path.join(path, 'LDAOutput', 'GensimLDA' + scale + '.csv'))

## look at output to see which models seem to best represent mobility or social exclusion
## mobility: municipal 8
## social: provincial 9
def model_score(text, scale, model_no):
  clean_text = dictionary.doc2bow(text)
  results = scale_models[scale][clean_text]
  if model_no in [x[0] for x in results]:
    return [x[1] for x in results if x[0] == model_no][0]
  else:
    return None

# all_text_docs['mobility_docs_score'] = all_text_docs['Gensim_processed'].apply(model_score, args=('Municipal', 9,))
# all_text_docs['social_docs_score'] = all_text_docs['Gensim_processed'].apply(model_score, args=('Provincial', 9,))
# plt.close()
# all_text_docs['mobility_docs_score'].hist(figsize=(4,4), bins=20)
# plt.title("Mobility")
# plt.xlabel("Probability Document Belongs to Topic")
# plt.ylabel("Frequency")
# plt.savefig(os.path.join(path, 'LDAOutput', 'MobilityTopicHistogram.png'))
# plt.close()
# all_text_docs['social_docs_score'].hist(figsize=(4,4), bins=20)
# plt.title("Social")
# plt.xlabel("Probability Document Belongs to Topic")
# plt.ylabel("Frequency")
# plt.savefig(os.path.join(path, 'LDAOutput', 'SocialTopicHistogram.png'))

## write out titles of Documents
# all_text_docs.loc[all_text_docs['mobility_docs_score'] > 0.5, ['Location', 'Scale', 'mobility_docs_score']].to_csv(os.path.join(path, 'LDAOutput', 'MobilityTitles.csv'))
# all_text_docs.loc[all_text_docs['social_docs_score'] > 0.5, ['Location', 'Scale', 'social_docs_score']].to_csv(os.path.join(path, 'LDAOutput', 'SocialTitles.csv'))

## remove proper nouns as stopwords
def remove_proper_nouns(text):
  return [x for x in text if x not in proper_nouns]

def calculate_similarity(row, args):
  row_mean = []
  if args == 'mobility':
    terms = mobility_terms
  else:
    terms = social_terms
  return model.wv.n_similarity(row, terms)

federal_words = ['canada', 'nrtee', 'canadaos', 'canadians', 'canadian']
provincial_words = ['alberta', 'british', 'columbia', 'saskatchewan', 'manitoba', 'ontario', 'quebec', 'nova', 'scotia', 'brunswick', 'pei', 'prince', 'edward', 'newfoundland', 'manitobaos', 'manitobans', 'labrador', 'churchill', 'island', 'saskatchewanos', 'ontarioos', 'provinces', 'scotiaos', 'albertans', 'scotians', 'columbians']
municipal_words = ['edmonton', 'calgary', 'toronto', 'winnipeg', 'halifax', 'transformto', 'ottawa', 'edmontonos', 'cityos', 'calgaryians', 'hamilton', 'vancouver', 'calgarians', 'vancity', 'city']
proper_nouns = federal_words + provincial_words + municipal_words + french_words
no_proper_nouns = [remove_proper_nouns(x) for x in all_processed_docs]

# set up Word2Vec
mobility_terms = ['mobility', 'transportation', 'transit', 'accessibility', 'bike', 'walk']
social_terms = ['social', 'family', 'friends', 'wellbeing', 'entertainment', 'health', 'mental', 'community']
model = Word2Vec(no_proper_nouns , size=250, window=10, min_count=10, workers=8)
# test similarity of individual words
# model.wv.similar_by_word('family')

# perform LDA and rate topics using Word2Vec
# first create all models for testing
word_columns = ['Word_' + str(x) for x in range(10)]
no_belows = [0, 1, 2, 5, 10, 15, 20]
no_aboves = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
for no_below in no_belows:
  for no_above in no_aboves:
    dictionary = gensim.corpora.Dictionary(no_proper_nouns)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=10000)
    for scale in scales:
      try:
        gensim.models.LdaMulticore.load(os.path.join(path, 'LDAOutput', 'Models', 'NPNGensimLDA' + scale + '_' + str(no_below) + '_' + str(no_above*10)))
      except:
        scale_docs = all_text_docs.loc[all_text_docs['Scale'] == scale, 'Gensim_processed']
        processed_docs = [remove_proper_nouns(x) for x in scale_docs]
        bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
        scale_models[scale] = gensim.models.LdaMulticore(bow_corpus, num_topics=topic_count, id2word=dictionary, passes=2, workers=6, random_state=10)
        scale_models[scale].save(os.path.join(path, 'LDAOutput', 'Models', 'NPNGensimLDA' + scale + '_' + str(no_below) + '_' + str(no_above*10)))

# test model score using perplexity DOES WORK WELL
# dictionary = gensim.corpora.Dictionary(no_proper_nouns)
# dictionary.filter_extremes(no_below=max(no_belows), no_above=min(no_aboves), keep_n=10000)
# bow_corpus = [dictionary.doc2bow(doc) for doc in no_proper_nouns]
# chunk = sample(bow_corpus, int(len(bow_corpus)*0.01))
# scale_models[scale].log_perplexity(chunk)
#
DF = {}
# for scale in scales:
#   DF[scale] = DataFrame(index=no_belows, columns=no_aboves)
#
# for no_below in no_belows:
#   for no_above in no_aboves:
#     for scale in scales:
#       lda = gensim.models.LdaMulticore.load(os.path.join(path, 'LDAOutput', 'Models', 'NPNGensimLDA' + scale + '_' + str(no_below) + '_' + str(no_above*10)))
#       LL = lda.log_perplexity(chunk)
#       # print("Scale={}, no_below={}, no_above={}, LL={:.2f}".format(scale, no_below, no_above, LL))
#       DF[scale].loc[no_below, no_above] = LL # higher LL is better
# for scale in scales:
#   print(scale)
#   print(DF[scale])

# test using the similarity of the test words
multiindex = pd.MultiIndex.from_product([scales, no_belows], names=['Scale', 'No Below'])
for item in ['mobility', 'social']:
  DF[item] = DataFrame(index=multiindex, columns=no_aboves)

for no_below in no_belows:
  for no_above in no_aboves:
    for scale in scales:
      lda = gensim.models.LdaMulticore.load(os.path.join(path, 'LDAOutput', 'Models', 'NPNGensimLDA' + scale + '_' + str(no_below) + '_' + str(no_above*10)))
      topics = lda.show_topics(formatted=False)
      topicsDF = DataFrame(index=range(topic_count), columns=word_columns)
      for i in range(topic_count):
        for j in range(10):
          topicsDF.iloc[i,j] = topics[i][1][j][0]
      for item in ['mobility', 'social']:
        topicsDF[item + '_similarity'] = topicsDF[word_columns].apply(calculate_similarity, axis=1, args=[item])
        DF[item].loc[(scale, no_below), no_above] = topicsDF[item + '_similarity'].max()
      #topicsDF.to_csv(os.path.join(path, 'LDAOutput', 'NPNGensimLDA' + scale + '.csv'))
for item in ['mobility', 'social']:
  print(item)
  print(DF[item])
  DF[item].to_csv(os.path.join(path, 'LDAOutput', 'Sensitivity', 'NPN' + item + 'SensitivityScores.csv'))

# load best model, calculate similarity, and write out tables
no_below = 5
no_above = 0.4
topicsDF = {}
for scale in scales:
  lda = gensim.models.LdaMulticore.load(os.path.join(path, 'LDAOutput', 'Models', 'NPNGensimLDA' + scale + '_' + str(no_below) + '_' + str(no_above*10)))
  topics = lda.show_topics(formatted=False)
  topicsDF = DataFrame(index=range(topic_count), columns=word_columns)
  for i in range(topic_count):
    for j in range(10):
      topicsDF.iloc[i,j] = topics[i][1][j][0]
  for item in ['mobility', 'social']:
    topicsDF[item + '_similarity'] = topicsDF[word_columns].apply(calculate_similarity, axis=1, args=[item])
  for i in range(topic_count): # redo the cells to include likelihood score
    for j in range(10):
      topicsDF.iloc[i,j] = topics[i][1][j][0] + '\n({:.4f})'.format(topics[i][1][j][1])
  topicsDF.index = ['Topic_' + str(x) for x in topicsDF.index]
  topicsDF.to_csv(os.path.join(path, 'LDAOutput', 'FinalModel', 'NPNGensimLDA' + scale + '.csv'))

## check similarity scores of outputs for best representtion of mobility or social exclusion
## mobility: municipal 6
## social: municipal 2
# load models again as scale_models since model_score requires that
scale_models = {}
for scale in scales:
  scale_models[scale] = gensim.models.LdaMulticore.load(os.path.join(path, 'LDAOutput', 'Models', 'NPNGensimLDA' + scale + '_' + str(no_below) + '_' + str(no_above*10)))
# the model_score function need the dicitonary so recreate it with proper parameters
dictionary = gensim.corpora.Dictionary(no_proper_nouns)
dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=10000)
# score all documents for their probability of belonging to a topic
all_text_docs['npn_mobility6_docs_score'] = all_text_docs['Gensim_processed'].apply(model_score, args=('Municipal', 6,))
all_text_docs['npn_social_docs_score'] = all_text_docs['Gensim_processed'].apply(model_score, args=('Municipal', 2,))

## write out titles of Documents
all_text_docs.loc[all_text_docs['npn_mobility7_docs_score'] > 0.5, ['Location', 'Scale', 'npn_mobility7_docs_score']].to_csv(os.path.join(path, 'LDAOutput', 'NPNMobility7Titles.csv'))
all_text_docs.loc[all_text_docs['npn_social_docs_score'] > 0.5, ['Location', 'Scale', 'npn_social_docs_score']].to_csv(os.path.join(path, 'LDAOutput', 'NPNSocialTitles.csv'))
plt.close()
all_text_docs['npn_mobility_docs_score'].hist(figsize=(4,4), bins=20)
plt.title("Mobility\n(without proper nouns)")
plt.xlabel("Probability Document Belongs to Topic")
plt.ylabel("Frequency")
plt.savefig(os.path.join(path, 'LDAOutput', 'NPNMobilityTopicHistogram.png'))
plt.close()
all_text_docs['npn_social_docs_score'].hist(figsize=(4,4), bins=20)
plt.title("Social\n(without proper nouns)")
plt.xlabel("Probability Document Belongs to Topic")
plt.ylabel("Frequency")
plt.savefig(os.path.join(path, 'LDAOutput', 'NPNSocialTopicHistogram.png'))
plt.close()

## check docs which weren't selected
## only check municipal documents since those scored highest
all_text_docs.loc[(all_text_docs['npn_mobility_docs_score'] < 0.05) & (all_text_docs['Scale'] == 'Municipal'), 'Location']
all_text_docs.loc[(all_text_docs['npn_social_docs_score'].isnull()) & (all_text_docs['Scale'] == 'Municipal'), 'Location'].sample(n=5)
