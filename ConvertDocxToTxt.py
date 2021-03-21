import docx
import os

path = '/Users/darcy/University/UofA/PhD/EAS/PhDWork/Studies/1 Policy/Documents'

## now read in all Word documents
def getText(filename):
  doc = docx.Document(filename)
  fullText = []
  for para in doc.paragraphs:
    fullText.append(para.text)
  return '\n'.join(fullText)

for dirpath, dirnames, files in os.walk(path):
  for name in files:
    if name.lower().endswith('docx'):
      out_name = name.replace('docx', 'txt')
      with open(os.path.join(path, 'ExtractedText', out_name), 'w') as f:
        f.write(getText(os.path.join(dirpath, name)))
