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
udo = Node(2)
marc = Node(1, parent=udo)
lian = Node("Liadazdan", parent=marc)
dan = Node("Dadazdazn", parent=udo)
jet = Node("Jedazdazt", parent=dan)
jan = Node("Jadddn", parent=dan)
joe = Node("Jdazdzaoe", parent=dan)

L = [udo, marc]
print(L)
# print(marc)
# Node('/Udo')
# print(joe)
# Node('/Udo/Dan/Joe')

# for pre, fill, node in RenderTree(udo):
#   print("%s%s" % (pre, node.name))

from anytree.exporter import DotExporter
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
def leftArc(S, I, A):
    A.append([I[len(I)-1], S.pop()]) # add the arc (j,i) to A
    # S.pop() # and pop the stack
    return 0
# the parameter I is not used in leftArc it seems

def rightArc(S, I, A):
    A.append([S[len(S)-1], I[len(I)-1]]) # Adds an arc from wi to wj from the token wi on top of the stack to the next input token wj
    S.append(I[len(I)-1]) # Pushes wj onto S.
    return 0

def reduce(S):
    S.pop()
    return S

def shift(S, I):
    # smth = remove(I[0])
    S.append(I.pop())
    return S

'''
Input: A sentence
Output: The dependency parser of the sentence as a parsing tree
'''
def NivreParser(sentence):
    # Initialisation of the parser configuration
    sentence.reverse() # to transform into a stack
    # print(sentence)
    n = len(sentence)
    S = [0] # We start with root on the stack
    I = [k for k in range(n)]
    A = [] # arcs

    while (len(I) != 0):
        # if len(S) == 0:
        #     shift(S, I)

        topStack = S[len(S)-1]
        topBuffer = I[len(I)-1]
        noAction = True
        # for i in range(len(A)):
        #     # if A[i][0] == topBuffer and A[i][1] == topStack:
        #         leftArc(S, I, A)
        #         noAction = False
        #         break
        #     elif A[i][0] == topStack and A[i][1] == topBuffer:
        #         hasChildren = False
        #         for j in range(len(A)):
        #             if A[j][0] == topBuffer:
        #                 hasChildren = True
        #         if hasChildren == True:
        #             rightArc(S, i, A)
        #             noAction = False
        #         break
        #     elif A[i][1] == topStack:
        #         reduce(S)
        #         noAction

        hasParent = False
        for i in range(len(A)):
            if A[i][1] == topStack:
                hasParent = True
                break
        if hasParent == False:
            leftArc(S, I, A)
            noAction = False

        if noAction == True:
            hasParent = False
            for i in range(len(A)):
                if A[i][1] == topBuffer:
                    hasParent = True
                    break
            if hasParent == False:
                rightArc(S, I, A)
                noAction = False

        if hasParent == True:
            reduce(S)

        shift(S, I)

    return A

def main():
    data = "My name is Nivre from South Korea."
    sentences = pre_process(data)

    print(sentences, end = "\n\n")

    dictionary = create_dictionary(sentences)
    int2word, word2int = dictionary
    vocab_size = len(int2word)
    print("Number of unique words = ", vocab_size, end = "\n")
    # print(dictionary, end = "\n\n")

    s1 = sentences[0]
    print(s1)

    # Nivre Parser
    parser = NivreParser(s1)
    print(parser)


    # Creation tree -> to review
    tree = []
    for i in range(len(s1)):
        tree.append(Node(s1[i]))

    for i in range(len(parser)):
        tree[parser[i][1]] = Node(s1[parser[i][1]], parent = tree[parser[i][0]])
    # print(tree)

    # for pre, fill, node in RenderTree(tree[len(tree)-1]):
    #     print("%s%s" % (pre, node.name))
    DotExporter(tree[len(tree)-1]).to_picture("tree.png")


if __name__ == '__main__':
    main()
