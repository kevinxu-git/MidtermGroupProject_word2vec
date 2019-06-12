# NLP Final Group Project : Dependency parser of Nivre for the Korean Language version 
# Yonsei University
# Groupe 8

import matplotlib.pyplot as pyplot
import random as r
import numpy as np
from math import *

# Trees - print trees
from anytree import Node, RenderTree
udo = Node("Uddazdazo")
marc = Node("Madzadazrc", parent=udo)
lian = Node("Liadazdan", parent=marc)
dan = Node("Dadazdazn", parent=udo)
jet = Node("Jedazdazt", parent=dan)
jan = Node("Jadddn", parent=dan)
joe = Node("Jdazdzaoe", parent=dan)

print(udo)
# Node('/Udo')
print(joe)
# Node('/Udo/Dan/Joe')

# for pre, fill, node in RenderTree(udo):	
# 	print("%s%s" % (pre, node.name))

# from anytree.exporter import DotExporter
# # graphviz needs to be installed for the next line!
# DotExporter(udo).to_picture("udo.png")




BAD_CHARS = [',', '<', '>', '!', '?', '-','<', ':',';','*','(',')','[',']','`','\'','"','는 ','은 ','과 ','이 ','그','저','가 ','을 ','를 ','에 ','와','나','로','의 ','도 ','께']

def has_digit(word):
    for char in word:
        if char.isdigit():
            return True
    return False

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

# Functions for arc-earger parsing algorithm
def leftArc():
  return 0
  
def rightArc():
  return 0
  
def reduce():
  return 0

def shift():
	return 0

def NivreParser(S, I, A):
	return 0

def main():
	data = "My name is Kevin. And I am in South Korea."
	sentences = pre_process(data)
	print(sentences, end = "\n\n")

	dictionary = create_dictionary(sentences)
	int2word, word2int = dictionary
	vocab_size = len(int2word)
	print("Number of unique words = ", vocab_size, end = "\n")
	print(dictionary, end = "\n\n")

	sentence1 = sentences[0]
	print(sentence1)




if __name__ == '__main__':
    main()

