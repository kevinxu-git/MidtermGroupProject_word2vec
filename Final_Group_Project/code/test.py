# NLP Final Group Project : Dependency parser of Nivre for the Korean Language version 
# Yonsei University
# Groupe 8

# 2019.06.18 3 p.m. Science Building 225

import matplotlib.pyplot as pyplot
import random as r
import numpy as np
from math import *

# Trees - print trees
from anytree import Node, RenderTree
udo = Node("Uddazdazo")
marc = Node(1, parent=udo)
lian = Node("Liadazdan", parent=marc)
dan = Node("Dadazdazn", parent=udo)
jet = Node("Jedazdazt", parent=dan)
jan = Node("Jadddn", parent=dan)
joe = Node("Jdazdzaoe", parent=dan)

print(marc)
# Node('/Udo')
print(joe)
# Node('/Udo/Dan/Joe')

# for pre, fill, node in RenderTree(udo):	
# 	print("%s%s" % (pre, node.name))

# from anytree.exporter import DotExporter
# # graphviz needs to be installed for the next line!
# DotExporter(udo).to_picture("udo.png")




BAD_CHARS = [',', '<', '>', '!', '?', '-','<', ':',';','*','(',')','[',']','`','\'','"','는 ','은 ','과 ','이 ','그','저','가 ','을 ','를 ','에 ','와','나','로','의 ','도 ','께']

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



# Functions for arc-earger parsing algorithm : leftArc, rightArc, reduce, shift
def leftArc(S, I, A, j):
	A.append([j,len(S)]) # add the arc (j,i) to A 
	S.pop()              # and pop the stack
	return 0
# the parameter I is not used in leftArc it seems
  
def rightArc(S, I, A, j):
	A.append([len(S),j])
	S.append(I.remove(j)) # what's really to be removed from I to put onto the top of the stack S ? 
	return 0              # probably not j depending on how we code
  
def reduce(S):
	S.pop()              # pop the stack --> only delete a word ??
	return 0

def shift(S, I, A):
	smth = remove(I[0])  # is I[0] the right 'next input token' to be removed ?
	S.append(smth)
	return 0
# parameter A not used

'''
Input: A sentence
Output: The dependency parser of the sentence as a parsing tree 
'''
def NivreParser(sentence):
	# Initialisation of the parser configuration
	n = len(sentence)
	S = [0] # root
	I = [k for k in range(1, n)]
	A = [] # arcs

	while (len(I) != 0):
		I.pop()
		
	return 0

def main():
	data = "My name is Kevin from South Korea."
	sentences = pre_process(data)
	print(sentences, end = "\n\n")

	dictionary = create_dictionary(sentences)
	int2word, word2int = dictionary
	vocab_size = len(int2word)
	print("Number of unique words = ", vocab_size, end = "\n")
	print(dictionary, end = "\n\n")

	s1 = sentences[0]
	print(s1)

	L = np.ones((vocab_size, vocab_size))
	print(L)

	# Nivre Parser
	NivreParser(s1)





if __name__ == '__main__':
    main()


