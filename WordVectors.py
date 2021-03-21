import os
import re
import string
import pandas as pd
from pandas import DataFrame
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter

path = '/Users/darcy/University/UofA/PhD/EAS/PhDWork/Studies/1 Policy/Documents'
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
terms = ['carbon', 'pricing', 'climate', 'greenhouse', 'backstop', 'infrastructure', 'fuel', 'gas', 'transportation', 'transit', 'mobility', 'pass', 'social', 'exclude', 'alone', 'friend', 'family']
scales = ['Federal', 'Provincial', 'Municipal']

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

# count each term
for term in terms:
  all_text_docs[term + 'Count'] = all_text_docs['clean_text'].apply(lambda x: Counter(x)[term])
term_countDF = DataFrame()
for scale in scales:
  term_countDF[scale] = all_text_docs.loc[all_text_docs['Scale'] == scale, [term + 'Count' for term in terms]].sum()
term_countDF.index = [x.replace('Count','') for x in term_countDF.index]
term_countDF.to_csv(os.path.join(path, 'TermCounts.csv'))

## try Word2Vec
sizes = range(50,550,50)
gensim_list = list(all_text_docs['clean_text'])
bigrams = Phraser(Phrases(gensim_list))
termsDF = DataFrame(index=pd.MultiIndex.from_product([terms, windows], names=['terms', 'sizes']), columns=["Term" + str(x) for x in range(1,11)])
for size in sizes:
  model = Word2Vec(bigrams[gensim_list] , size=size, window=10, min_count=10, workers=8)
#  model.train(gensim_list, total_examples=len(gensim_list), epochs=10)
  for term in terms:
    try:
      termsDF.loc[(term, size)] = [i for (i,v) in model.wv.most_similar(positive=term)]
    except:
      print("{} not in vocabulary".format(term))
termsDF.to_csv(os.path.join(path, 'Word2VecOutput', 'AllTermsDF.csv'))

# repeat but use scales
scales = ['Municipal', 'Provincial', 'Federal']
for scale in scales:
  gensim_list = list(all_text_docs.loc[all_text_docs['Scale'] == scale, 'clean_text'])
  bigrams = Phraser(Phrases(gensim_list))
  termsDF = DataFrame(index=pd.MultiIndex.from_product([terms, windows], names=['terms', 'windows']), columns=["Term" + str(x) for x in range(1,11)])
  for size in sizes:
    model = Word2Vec(bigrams[gensim_list] , size=size, window=10, min_count=10, workers=8)
#    model.train(gensim_list, total_examples=len(gensim_list), epochs=10)
    for term in terms:
      try:
        termsDF.loc[(term, size)] = [i for (i,v) in model.wv.most_similar(positive=term)]
      except:
        print("{} not in vocabulary".format(term))
  termsDF.to_csv(os.path.join(path, 'Word2VecOutput', scale + 'TermsDF.csv'))

# create visualization of terms
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
