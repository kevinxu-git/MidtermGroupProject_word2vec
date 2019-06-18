# NLP Final Group Project : Dependency parser of Nivre for the Korean Language version 
# Yonsei University
# Groupe 8

# 2019.06.18 3 p.m. Science Building 225

import matplotlib.pyplot as pyplot
import random as r
import numpy as np
from math import *

# Library for trees
from anytree import Node, RenderTree
from anytree.exporter import DotExporter # graphviz needs to be installed for the next line


# # Help function for pre_process()
# def has_digit(word):
#     for char in word:
#         if char.isdigit():
#             return True
#     return False

# '''
# Input: textfile in form of string
# Output: List of list of words [[sentence1], [sentence2],...]
# Side-effect: Removes some unwanted characters and words with numbers.
# '''
# def pre_process(text):
#     filtered = ""
#     for char in text:
#         if char not in BAD_CHARS:
#             if char == '.': # Splits '.' from sentence but still want to know where sentence ends.
#                 char = " " + char
#             filtered = filtered + char
#     filtered_list = filtered.split()
#     sentences = []
#     sentence = []
#     for word in filtered_list:
#         if not has_digit(word):
#             if word == '.':
#                 sentences.append(sentence)
#                 sentence = []
#             else:
#                 sentence.append(word)
#     return sentences

# '''
# Input: List of sentences
# Output: Look-up tables for converting a word to int and vice versa.
# '''
# def create_dictionary(list_of_sentences):
#     words = []
#     for sentence in list_of_sentences:
#         for word in sentence:
#             if word not in words:
#                 words.append(word)

#     int2word = {}
#     word2int = {}
#     for i, word in enumerate(words):
#         word2int[word] = i
#         int2word[i] = word

#     return int2word, word2int



# Functions for arc-earger parsing algorithm : leftArc, rightArc, reduce, shift
def leftArc(S, I, A):
    A.append([I[len(I)-1], S.pop()]) # Add the arc (j,i) to A 
    # S.pop() # Pop the stack
    return 0
  
def rightArc(S, I, A):
    A.append([S[len(S)-1], I[len(I)-1]]) # Add an arc from wi to wj from the token wi on top of the stack to the next input token wj
    S.append(I[len(I)-1]) # Push wj onto S.
    return 0              
  
def reduce(S):
    S.pop() # Pop the stack S
    return S

def shift(S, I):
    S.append(I.pop()) # Push the top token of the stack I on the stack S and pop the stack I
    return S

'''
Input: A sentence
Output: The dependency parser of the sentence as a parsing tree 
'''
def NivreParser(sentence):
    # Initialisation of the parser configuration
    n = len(sentence)
    S = [0] # Start with root on the stack
    I = [k for k in range(n)]
    A = [] # Arcs

    while (len(I) != 0):
        topStack = S[len(S)-1]
        topBuffer = I[len(I)-1]
        noAction = True

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

'''
Input: A the arcs in the tree and the sentence 
Output: Generate a PNG file of the tree
'''
def printTree(A, sentence):
    # Creation tree -> to review
    print(sentence)
    tree = [Node("root")]
    for i in range(len(sentence)):
        tree.append(Node(sentence[i]))
    tree[1] = Node(sentence[A[0][0]], parent=tree[0])

    for i in range(len(A)):
        tree[A[i][1]+1] = Node(sentence[A[i][1]], parent = tree[A[i][0]+1]) 

    print(tree)
    DotExporter(tree[0]).to_picture("tree.png")
    return 0

def main():
    data = "몇 시가 되었는지 경지는 모르고 있었다 ."

    # sentences = pre_process(data)

    # print(sentences, end = "\n\n")

    # dictionary = create_dictionary(sentences)
    # int2word, word2int = dictionary
    # vocab_size = len(int2word)
    # print("Number of unique words = ", vocab_size, end = "\n")
    # # print(dictionary, end = "\n\n")

    s1 = data.split()
    print(s1)

    # Nivre Parser
    parser = NivreParser(s1)
    parser = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
    print(parser)

    printTree(parser, s1)

if __name__ == '__main__':
    main()


