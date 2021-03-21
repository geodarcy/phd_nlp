import os
from bs4 import BeautifulSoup

path = '/Users/darcy/University/UofA/PhD/EAS/PhDWork/Studies/1 Policy/Documents'

files = os.listdir(os.path.join(path, 'Webpages'))

for file in files:
  with open(os.path.join(path, 'Webpages', file), 'r') as f:
    soup = BeautifulSoup(f, 'html.parser')
  text = [x.text.strip() for x in soup.find_all('p')]
  with open(os.path.join(path, 'WebpagesProcessed', file), 'w') as wf:
    for item in text:
      wf.write("{}\n".format(item))

## some special cases
file = os.path.join(path, 'Webpages/Q-2, r. 46.1 - Regulation respecting a cap-and-trade system for greenhouse gas emission allowances.txt')
with open(os.path.join(path, 'Webpages', file), 'r') as f:
  soup = BeautifulSoup(f, 'html.parser')
text = [x.text.strip() for x in soup.find_all('span')]
with open(os.path.join(path, 'WebpagesProcessed/Q-2, r. 46.1 - Regulation respecting a cap-and-trade system for greenhouse gas emission allowances.txt'), 'w') as wf:
  for item in text:
    wf.write("{}\n".format(item))
