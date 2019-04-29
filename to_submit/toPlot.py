#-*- coding:utf-8 -*-

# gensim
import gensim.downloader as api
# Korean corpus
from konlpy.corpus import kolaw # imports the textfile pointer
from konlpy import utils

import matplotlib.pyplot as pyplot
import random as r
import numpy as np
from math import *
import codecs

import matplotlib.font_manager as fm

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


BAD_CHARS = [',', '<', '>', '!', '?', '-','<', ':',';','*','(',')','[',']','`','\'','"','는 ','은 ','과 ','이 ','그','저','가 ','을 ','를 ','에 ','와','나','로','의 ','도 ','께']

WINDOW_SIZE = 2

# Help function for pre_process()
def has_digit(word):
    for char in word:
        if char.isdigit():
            return True
    return False

'''
Input: textfile in form of string
Output: List of list of words [[sentence1], [sentence2],...]
Side-effect: Removes some unwanted characters and words with numbers.
'''
def pre_process(text):
    filtered = ""
    for char in text:
        if char not in BAD_CHARS:
            if char == '.': # Splits '.' from sentence but still want to know where sentence ends.
                char = " " + char
            filtered = filtered + char
    filtered_list = filtered.split()
    sentences = []
    sentence = []
    for word in filtered_list:
        if not has_digit(word):
            if word == '.':
                sentences.append(sentence)
                sentence = []
            else:
                sentence.append(word)
    return sentences

'''
Input: List of sentences
Output: Look-up tables for converting a word to int and vice versa.
'''
def create_dictionary(list_of_sentences):
    words = []
    for sentence in list_of_sentences:
        for word in sentence:
            if word not in words:
                words.append(word)

    int2word = {}
    word2int = {}
    for i, word in enumerate(words):
        word2int[word] = i
        int2word[i] = word

    return int2word, word2int


U = np.loadtxt("matrix_U.txt")
V = np.loadtxt("matrix_V.txt")

f = open("financenews.txt", "r", encoding = "utf-8")
data = f.read()
f.close()
sentences = pre_process(data)


# Clean up and pre-process
dictionary = create_dictionary(sentences)
int2word, word2int = dictionary
vocab_size = len(int2word)
print("Number of unique words = ", vocab_size, end = "\n")
# print("Dictionary : ", end = "\n")
# print(list(word2int), end = "\n\n")

# PCA
X = U+V
pca = PCA(n_components = 2)
result = pca.fit_transform(X.transpose())
# print("Explained variance ratio : " ,pca.explained_variance_ratio_, end = "\n")
# print("Singular values : ",pca.singular_values_, end = "\n")

# Plot of word vectors
font_location = 'Typo_DodamM.ttf'
# ex - 'C:/asiahead4.ttf'
prop = fm.FontProperties(fname = font_location)

pyplot.scatter(result[:, 0], result[:, 1])

for i in range(len(U)):
    pyplot.annotate(int2word[i], xy=(result[i, 0], result[i, 1]), fontproperties = prop)
pyplot.xlabel(u'언녕', fontproperties = prop)
pyplot.show()