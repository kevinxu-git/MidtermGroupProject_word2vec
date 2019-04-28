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

BAD_CHARS = [',', '<', '>', '!', '?', '-','<', ':',';','*']

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

'''
Input: sentences in list-form and dictionary 'word2int'
Output: Batch containing list of all [[center_word, neighbour1], [cw, n2],..]
        represented as integers.
'''
def generate_batch(window_size, sentences, word2int):
    batch = []
    for i, sentence in enumerate(sentences):
        sentence_length = len(sentence)
        for i, word in enumerate(sentence):
            for neighbour in sentence[max(i - window_size, 0) : min(i + window_size + 1, sentence_length + 1)]:
                if neighbour != word:
                    batch.append([word2int[word], word2int[neighbour]])
    return batch

'''
Input: the size of the vector and the index of the 1
Output: a one-hot vector
'''
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros((vocab_size, 1))
    temp[data_point_index] = 1
    return temp

'''
Input: c the index of center of, U and V two square matrices
Output: the predicted distribution of the outside words of the center word c
'''
def y_hat(c, U, V):
	x = np.dot(U, V[:, c])
	tmp = np.max(x)
	x -= tmp
	x = np.exp(x)
	tmp = np.sum(x)
	x /= tmp
	return x

'''
Input: the parameters of the loss J
Output: the partial derivative of J with respect to v
'''
def J_ns_deriv_v(c, o, U, y, y_hat):
	return U.dot(y_hat - y).transpose()[0]

'''
Input: the parameters of the loss J
Output: the partial derivative of J with respect to u
'''
def J_ns_deriv_u(c, o, U, V, w, y, y_hat):
	if w == o:
		return V[:, c] * y_hat[o] - V[:, c] * y[o]
	return V[:, c] * y[w]

'''
Input: theta the parameter for the stochastic gradient descent
Output: U and V matrices of word2vec
'''
def U_V_from_theta(theta):
    split = np.vsplit(theta, 2)
    return split[1].transpose(), split[0].transpose()

'''
Input: c the index of the center word, theta the parameter of stochastic gradient descent, batch the context window and start index of the context window
Output: the gradient of J and the index of the next context window in the batch
'''
def grad_J(c, theta, batch, start):
    vocab_size = len(theta[0])
    grad = np.zeros_like(theta)
    # U and V
    U, V = U_V_from_theta(theta)
    # start is the indice of the mini batch we are working on.
    for i in range(start, len(batch)):
        if c == batch[i][0] :
        	j = batch[i][1]
        	y = to_one_hot(j, vocab_size)
        	y_h = y_hat(c, U, V)

        	# V part of grad J
        	grad[c, :] += J_ns_deriv_v(c, j, U, y, y_h)

        	# U part of grad J
        	for w in range(vocab_size):
        		grad[vocab_size+w, :] += J_ns_deriv_u(c, j, U, V, w, y, y_h)

        # If the next center word is not the center word in the batck we are working on :
        # then we stop. return the vector grad and the indice of the next mini batch
        if i+1 <= len(batch) - 1 :
            if c != batch[i+1][0]:
            	return grad, i+1
    return grad, i+1

def word2vec_skip_gram(dictionary, sentences):
    int2word, word2int = dictionary
    vocab_size = len(int2word)

	# Generate batch
    batch = generate_batch(WINDOW_SIZE, sentences, word2int)
    print("Number of batches = ", len(batch), end='\n\n')

    # Initialization of U and V
    U = np.zeros((vocab_size, vocab_size))
    V = np.zeros((vocab_size, vocab_size))

    for i in range(vocab_size):
        for j in range(vocab_size):
            U[i,j]=r.uniform(0,1)
            V[i,j]=r.uniform(0,1)

    # Stochastic gradient descent
    theta = np.vstack((V.transpose(), U.transpose()))
    alpha = 0.1
    i = 0
    while i != len(batch):
    	theta_grad, i = grad_J(batch[i][0], theta, batch, i)
    	theta = theta - alpha * theta_grad

    U, V = U_V_from_theta(theta)
    return U, V

def main():
    # Import data
    sentences = pre_process("Je m'appelle Kevin. Il s'appelle Eric. Je suis Eric. Je m'appelle Kevin. Je m'appelle Kevin. Je m'appelle Kevin. Je m'appelle Kevin. Je m'appelle Kevin. Je m'appelle Kevin. Je m'appelle Kevin. Je m'appelle Kevin.")
    print("Sentences : ")
    print(sentences)

    # Korean corpus
    # data = kolaw.open('constitution.txt').read()
    # sentences = pre_process(data)

    # gensim
    # corpus = api.load('text8')
    # data = " ".join(list(corpus)[0])
    # sentences = pre_process(data)
    # print(data)

    # f = open("financenews.txt", "r", encoding = "utf-8")
    # data = f.read()
    # f.close()
    # sentences = pre_process(data)


    # Clean up and pre-process
    dictionary = create_dictionary(sentences)
    int2word, word2int = dictionary
    vocab_size = len(int2word)
    print("Number of unique words = ", vocab_size, end = "\n")
    # print("Dictionary : ", end = "\n")
    # print(list(word2int), end = "\n\n")

    # To display Korean words
    # font_location = 'Typo_DodamM.ttf'
    # # ex - 'C:/asiahead4.ttf'
    # prop = fm.FontProperties(fname = font_location)
    # print(font_name)
    # pyplot.rc('font', family = prop)
    # pyplot.rc('font', **{'sans-serif' : 'Arial',
    #                      'family' : 'sans-serif'})

    # word2vec
    U, V = word2vec_skip_gram(dictionary, sentences)

    # Save U and V
    f = open("output.txt", "w")
    f.write("U = \n")
    f.write(str(U))
    f.write("\nV = \n")
    f.write(str(V))
    f.close()
    
    # PCA
    X = U+V
    pca = PCA(n_components = 2)
    result = pca.fit_transform(X.transpose())
    # print("Explained variance ratio : " ,pca.explained_variance_ratio_, end = "\n")
    # print("Singular values : ",pca.singular_values_, end = "\n")

    # Plot of word vectors
    pyplot.scatter(result[:, 0], result[:, 1])

    for i in range(len(U)):
            pyplot.annotate(int2word[i], xy=(result[i, 0], result[i, 1]))
    # pyplot.xlabel(u'언녕', fontproperties = prop)
    pyplot.show()

if __name__ == '__main__':
    main()
