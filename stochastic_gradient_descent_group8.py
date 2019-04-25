from matplotlib.pyplot import *
import random as r
import numpy as np
from math import *

# Import of other Python files
from sigmoid_function_group8 import *

# print(sigmoid(1))

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


def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

'''
Input: batch of word couples [[center_word, neighbour1], [cw, n2],..]
Output: One hot vectors of batch items as numpy arrays.
'''
def create_training_vectors(batch, dict_size):
    x_train = [] # input word
    y_train = [] # output word
    for item in batch:
        x_train.append(to_one_hot(item[0], dict_size))
        y_train.append(to_one_hot(item[1], dict_size))
        # print(x_train)
        # print(y_train)
        # convert them to numpy arrays
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    assert not np.any(np.isnan(x_train))
    assert not np.any(np.isnan(y_train))
    return x_train.transpose(), y_train.transpose()

def J_ns(c, o, U):
	return -log(U[o, c])

def J_ns_deriv_v(c, o, U, y):
	#print(U[:,c] - y)
	#print(U.transpose())
	return U.dot(U[:,c] - y)  # U*(y^ - y)

def J_ns_deriv_u(c, o, U, y, V, w):
	if w == o:
		return V[:, c] * U[o, c] - V[:, c] * y[o]
	return V[:, c] * y[w]

def J_sg(c, U, V, word_index):
	S = 0
	for j in word_index:
		S += J_ns(c, j, U)
	return S

def grad_J(c, theta, batch, start):
    vocab_size = len(theta[0])

    # U and V
    split = np.vsplit(theta, 2)
    V = split[0].transpose()
    U = split[1].transpose()

    grad = np.zeros_like(theta)
    #print(grad)

    # start is the indice of the mini batch we are working on.
    for i in range(start, len(batch)):
        if c == batch[i][0] :
        	j = batch[i][1]
        	y = to_one_hot(j, vocab_size)

        	# V part of grad J
        	grad[c, :] += J_ns_deriv_v(c, j, U, y)
        	# U part of grad J
        	for w in range(vocab_size):
        		grad[vocab_size+w, :] += J_ns_deriv_u(c, j, U, y, V, w)

        # if the next center word is not the center word in the batck we are working on :
        # then we stop. return the vector grad and the indice of the next mini batch
        if i+1 <= len(batch) - 1 :
            if c != batch[i+1][0]:
            	print(i)
            	return grad, i
    return grad, i

def main():
    #Import data, clean up and structure
    sentences = pre_process("He is here. She is there lol.")
    print("Sentences : ")
    print(sentences)

    int2word, word2int = create_dictionary(sentences)
    dictionary = create_dictionary(sentences)
    print("\nDictionary : ")
    print(dictionary)
    print(list(dictionary[1]))
    print('is' in dictionary[1])


    #Generate batch
    batch = generate_batch(WINDOW_SIZE, sentences, word2int)
    print("\nbatch")
    print(batch[1][1])
    print(len(batch))

    # Create one hot vectors
    dict_size = len(int2word)
    x_train, y_train = create_training_vectors(batch, dict_size)

    print("\nnb of words : ")
    print(len(int2word))

    print("\nV = ")
    print(x_train)
    print("\n y true empirical distribution = ")
    print(y_train)

    print(J_ns_deriv_v(1, 2, np.eye(6, 6), y_train[:, 1]))
    print(J_ns_deriv_u(1, 2, np.eye(6, 6), y_train[:, 1], x_train, 2))
    print(J_ns_deriv_u(0, 2, np.eye(6, 6), y_train[:, 1], x_train, 0))


    # the context windows for center word c
    c = 1
    context_windows = y_train[:, 1:3]
    # print(np.where(context_windows[:,1] == 1)[0][0])

    # The words index in the context windows
    words_index = []
    for i in range(len(context_windows[0])):
    	words_index.append(np.where(context_windows[:,i] == 1)[0][0])
    print(words_index)

    print(J_sg(c, np.eye(6, 6)+2, np.eye(6, 6), words_index))

    voc_size = dict_size
    print("\nsize dico = ")
    print(voc_size)



    # SDG
    U = np.zeros((voc_size, voc_size)) + 1
    V = np.zeros((voc_size, voc_size)) + 1
    print("\nU = ")
    print(U)

    theta = np.vstack((V.transpose(), U.transpose()))
    print("\ntheta ini = ")
    print(theta)
    alpha = 0.1
    # while True:
    i = 0
    for z in range(20):
    	theta_grad, i = grad_J(batch[i][0], theta, batch, i)
    	theta = theta - alpha * theta_grad
    	print(z)

    print("\ntheta grad = ")
    print(theta)
    print("\ni=")
    print(i)



if __name__ == '__main__':
    main()


