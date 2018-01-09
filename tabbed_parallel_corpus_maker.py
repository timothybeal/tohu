# Takes all the English translations and matches them to Hebrew transliterated
# text with a tab between each Hebrew and English verse and a \n at the end of
# each verse pair to work with the simple keras model at https://blog.keras.io/
# a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html.

import re

tthandle = open("heb_corpus.txt", encoding="utf-8")
ttext = tthandle.read()
ttext = re.sub(' \.', '.', ttext)
ttext = re.sub('\n', '\t\n', ttext)
ttext = re.split('\n', ttext)

thandle = open("eng_corpus2.txt", encoding="utf-8")
text = thandle.read()
text = re.sub(' \.', '.', text)
text = re.split('\n', text)

heb_eng_corp = str()
verse = str()
verse_num = 0

for verse in ttext:
    verse = ttext[verse_num] + text[verse_num] + '\n'
    heb_eng_corp = heb_eng_corp + verse
    verse_num = verse_num + 1
    if verse_num > len(ttext):
        break

print(heb_eng_corp)

with open('tabbed_heb_eng_corpus2.txt', 'w', encoding='utf-8') as tt:
    tt.write(heb_eng_corp)
