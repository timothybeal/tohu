# A simple program for counting verses in each txt file to make sure all are the same length before
# combining them into a parallel corpus.

import re

with open('eng_corpus2.txt', encoding='utf-8') as f:
    text = f.read()
    scount = re.findall('\n', text)

print(len(scount))

with open('heb_corpus.txt', encoding='utf-8') as f:
    text = f.read()
    scount = re.findall('\n', text)

print(len(scount))
