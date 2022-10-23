# First, merge three data sources
from ast import List
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
class preProcessing():
    m={}
    l=[]
    line = []
    y = []
    words = []
    all_stopwords = stopwords.words('english')
    all_stopwords.append('.')
    all_stopwords.append(',')
    def merge(args):
        for f in args:
            preProcessing.readFile(f)

    def readFile(file):
        ps = PorterStemmer()
        with open(file) as f:
            for line in f:
               temp= re.split('\t', line)
               preProcessing.line.append(temp[0])
               preProcessing.y.append(temp[1])
               map[temp[0]] : temp[1]   
               words = word_tokenize(temp[0])
               for word in words:
                  preProcessing.l.append(ps.stem(word))
    def getWord(lines):
        ps = PorterStemmer()
        words = []
        for line in lines:
              wds =word_tokenize(line)  
              for word in wds:
                  words.append(ps.stem(word))
        return words
       
    def init_matrix(line, words):
        words = list(set(words))
        ps = PorterStemmer()
        rows, cols = (len(line),len(words))
        matrix = []
        for i in range(len(line)):
            sentence = line[i]
            tokens =  word_tokenize(sentence)
            row = [0]*cols
            for j in range(len(tokens)):
             if  not ps.stem(tokens[j]) in words: continue
             row[words.index(ps.stem(tokens[j]))] +=1
            
            matrix.append(row)  
        return matrix
    
    def top_k_words(words, k: int):
        #remove stopwords
        words_without_sw = [word for word in words if not word in preProcessing.all_stopwords]
        dict = {}
        for x in words_without_sw:
            if x in dict:
                dict[x] += 1
            else:
                dict[x] = 1
        res = sorted(dict, key=lambda x: (-dict[x], x))
        return res[:k]
