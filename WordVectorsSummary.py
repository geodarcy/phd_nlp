import os
import string
import pandas as pd
from pandas import DataFrame

path = '/Users/darcy/University/UofA/PhD/EAS/PhDWork/Studies/1 Policy/Documents'
terms = ['carbon', 'pricing', 'climate', 'greenhouse', 'backstop', 'infrastructure', 'fuel', 'gas', 'transportation', 'transit', 'mobility', 'pass', 'social', 'exclude', 'alone', 'friend', 'family']
scales = ['Municipal', 'Provincial', 'Federal']
columns=["Term" + str(x) for x in range(1,11)]

scales_summary_Dict = {}
for scale in scales:
  scales_summary_Dict[scale] = pd.read_csv(os.path.join(path, 'Word2VecOutput', scale + 'TermsDF.csv')).set_index('terms')

terms_summaryDF = {}
for term in terms:
  scalesDF = DataFrame()
  frames = [DataFrame(scales_summary_Dict[scale].loc[term, columns].apply(pd.Series.value_counts).sum(axis=1), columns=[scale]) for scale in scales]
  terms_summaryDF[term] = pd.concat(frames, axis=1, sort=True)
  terms_summaryDF[term] = terms_summaryDF[term][(terms_summaryDF[term][scales] > 1)].dropna(how='all')
  terms_summaryDF[term].to_csv(os.path.join(path, 'Word2VecOutput', term + 'Summary.csv'))
