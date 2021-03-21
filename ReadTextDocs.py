import os
import pandas as pd
from pandas import DataFrame

path = '/Users/darcy/University/UofA/PhD/EAS/PhDWork/Studies/1 Policy/Documents'
encodings = ['utf-8', 'utf-16', 'utf-32', 'latin_1', 'ascii', 'cp037', 'cp437']

# all_text_docs = DataFrame(columns=['path', 'encoding', 'doc_text'])
stop_all = 1
count = 0
for dirpath, dirnames, files in os.walk(os.path.join(path, 'ExtractedText')):
  for name in files:
    if name[-3:] == 'txt' and stop_all == 1 and name not in all_text_docs.index:
      for encoding in encodings:
        try:
          with open(os.path.join(dirpath, name), 'rb') as f:
            doc_text = f.read().decode(encoding=encoding)
          print(doc_text[:200])
          print("\nIs this legible? (y/n/a(bort))")
          legible = input()
          if legible == 'y':
            outtext = DataFrame({'path': dirpath, 'encoding': encoding, 'doc_text': doc_text}, index=[name])
            all_text_docs = pd.concat([all_text_docs, outtext])
            break
          if legible == 'n':
            pass
          if legible == 'a':
            stop_all = 0
            break
        except:
          count += 1
print("Exiting gracefully.")
all_text_docs.to_pickle(os.path.join(path, 'AllDocsPickle'))
