import os
import nltk
import glob
import math
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import *
from collections import Counter
from collections import defaultdict


## Merge all files together
os.chdir('/Users/Guang/Downloads/transcripts/')
read_files = glob.glob("*.txt")
with open("merge.txt", 'w') as outfile:
    for f in read_files:
        with open(f, 'r') as infile:
            outfile.write(infile.read())


## Calculate number of word tokens for the merged text
text = open('merge.txt').read()
tokens = nltk.word_tokenize(text)
print(len(tokens)) 
## First nswer to Q1 is: 293356


## The steps below are for text processing:
## The RegexpTokenizer helps remove special characters such as punctuations
regexTokenizer = RegexpTokenizer(r'[a-zA-Z_]+')
tokens_processed_1 = [word.lower() for word in regexTokenizer.tokenize(text)] 
## Remove stop words
stopwords = open('/Users/Guang/Downloads/stopwords.txt').read()
tokens_processed_2 = [word for word in tokens_processed_1 if not word in stopwords]
## Stemming by using Porter stemmer
stemmer = PorterStemmer()
tokens_processed_3 = [stemmer.stem(word) for word in tokens_processed_2]
## The number of word tokens after processing
token_num = len(tokens_processed_3)
print(len(tokens_processed_3))
## Second answer to Q1 is: 108161


## Create counter to calculate frequency of all unique words
counts = Counter(tokens_processed_3)
print(len(counts))
## Answer to Q2 is the length of counts : 7256


## The words that only occur once
words_occur_once = [word for word in counts if counts[word] == 1]
print(len(words_occur_once))
## Answer to Q3 is the length of words_occur_once: 2808


## Number of words per document
aveWordsPerDoc = token_num / 404
print(aveWordsPerDoc)
## Answer to Q4 is the aveWordsPerDoc: 267.72


## Pull out the Top 30 most frequent words
top30 = dict(counts.most_common(30))
## Create dataframe for the tf of top 30 words
table = pd.DataFrame.from_dict(top30, orient='index').reset_index() 
table.columns = ['term', 'tf']
## Get the terms(keys for the dataframe)
keys = table['term'].tolist()


## tf(weight)
tf_weight = {}
for key in keys:
	tf = table[table.term == key].tf.item()
	tf_weight[key] = round(math.log2(tf) + 1.0, 3)
table['tf(weight)'] = table['term'].map(tf_weight)

  
## The document frequency
df = defaultdict(int)
for f in read_files:
      with open(f, 'r') as infile:
      	   t = infile.read()
      	   # token = nltk.word_tokenize(t)
      	   token_1 = [word.lower() for word in regexTokenizer.tokenize(t)]
      	   token_2 = [word for word in token_1 if not word in stopwords]
      	   token_3 = [stemmer.stem(word) for word in token_2]
      	   for key in keys:
      	   	   if key in token_3:
      	   	   	  df[key] += 1
table['df'] = table['term'].map(df)


## idf
idf = {}
for key in keys:
	df = table[table.term == key].df.item()
	idf[key] = round(math.log2(404 / df), 3)
table['idf'] = table['term'].map(idf)


## tf*idf
tf_idf= {}
for key in keys:
	tf = table[table.term == key].tf.item()
	idf = table[table.term == key].idf.item()
	tf_idf[key] = round(tf * idf, 3)
table['tf_idf'] = table['term'].map(tf_idf)


## p(term)
p_term= {}
for key in keys:
	tf = table[table.term == key].tf.item()
	p_term[key] = round(tf / token_num, 5)
table['p_term'] = table['term'].map(p_term)


## The table is the answer for Q5
print(table)
table.to_csv('/Users/Guang/Downloads/finalTable', sep='\t')

