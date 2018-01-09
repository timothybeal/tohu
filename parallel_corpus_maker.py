# (1) Concatenates multiple translations into a single English corpus
# string, checks its verse length (marked by newlines), then writes the
# string to a text file ("eng_corpus.txt"); (2) concatenates an equal
# number of Hebrew Torah strings into a single parallel Hebrew corpus string,
# checks its verse length (marked by newlines), and writes the string to a
# text file ("heb_corpus.txt"); and (3) does the same as (2) to make a
# parallel corpus transliterated Hebrew ("translit_corpus.txt").

import re

# Clean, concatenate, and write the English corpus from multiple translations

eng_files = ['schocken_newlines.txt','youngs_newlines.txt', 'jps_newlines.txt',
            'geneva_newlines.txt', 'kjv_newlines.txt', 'douay_newlines.txt',
            'asb_newlines.txt', 'rv_newlines.txt','rsv_newlines.txt',
            'nrsv_newlines.txt']

num_files = len(eng_files)
eng_corpus = str()

for file in eng_files:
    fhandle = open(file, encoding='utf-8')
    text = fhandle.read()
#    text = re.sub(r' *([.,:;?!—]) *', r' \1 ', text)
#    text = re.sub(r' *([.?!]) *', r' \1 ', text)
    text = re.sub(r' *([,:;?!.—“”"`‘’…éωΩ\[\]\(\)\t]) *', r' ', text)
    text = re.sub("’s", "'s", text)
    text = re.sub('\n ', '. \n', text)
    # text = re.sub('\ufeff', '', text)
    # text = re.sub('\xa0', '', text)
    eng_corpus = eng_corpus + text

eng_versecount = re.findall('\n', eng_corpus)
print("Total verses in English corpus:", len(eng_versecount))

with open("eng_corpus2.txt", "w", encoding="utf-8") as t:
    t.write(eng_corpus)

# Compile and write the Hebrew corpus with as many repeats of itself
# as needed to parallel the number of translations in the English corpus.

fhandle = open('heb_newlines.txt', encoding='utf-8')
text = fhandle.read()
heb_corpus = text * num_files

heb_versecount = re.findall('\n', heb_corpus)
print("Total verses in Hebrew corpus: ", len(heb_versecount))

with open("heb_corpus.txt", "w", encoding="utf-8") as t:
    t.write(heb_corpus)

# Compile and write a transliterated version of the corpus, same as above.

fhandle = open('heb_translit_newlines.txt', encoding='utf-8')
ttext = fhandle.read()
ttext = re.sub('\n ', '\n', ttext)
translit_corpus = ttext * num_files

translit_versecount = re.findall('\n', translit_corpus)
print("Total verses in transliterated corpus: ", len(translit_versecount))

with open("translit_corpus.txt", "w", encoding="utf-8") as tt:
    tt.write(translit_corpus)
