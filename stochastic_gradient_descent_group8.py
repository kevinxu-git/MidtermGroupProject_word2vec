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
            for neighbour in sentence[max(i - window_size, 0) : min(i + window_size, sentence_length + 1)]:
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
    return x_train, y_train

def J_ns(v, o, U):
	return 1

def J_ns_deriv_v(v, o, U):
	return 1

def J_ns_deriv_u(v, o, U):
	return 1

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
    print(batch)

    # Create one hot vectors
    dict_size = len(int2word)
    x_train, y_train = create_training_vectors(batch, dict_size)

    print("\nnb of words : ")
    print(len(int2word))


    print(x_train)
    print(y_train)

if __name__ == '__main__':
    main()


