#-*- coding:utf-8 -*-

# NLP Final Group Project : Dependency parser of Nivre for the Korean Language version
# Yonsei University
# Groupe 8

# 2019.06.18 3 p.m. Science Building 225

import matplotlib.pyplot as pyplot
import random as r
import numpy as np
from math import *

import os
import codecs as c
from io import StringIO, BytesIO
from konlpy.tag import Kkma
from konlpy.utils import pprint

# Library for trees
from anytree import Node, RenderTree
from anytree.exporter import DotExporter # graphviz needs to be installed for the next line


def extract_body(data):
    body = ""
    save = False
    for i, char in enumerate(data):
        if char == '<':
            check = data[i:i+6]
            if check == "<body>":
                save = True
            if check == "</body":
                save = False
        if save and char != '\t':
            body = body  + char
    return body[6:]

def parse(body):
    # print(body)
    sentences = []
    sentence_add = False
    sentence = ""
    dictionary = []
    dict_add = False

    for i, char in enumerate(body):
        if char == '\r':
            if sentence_add:
                sentences.append(sentence)
                dictionary.append(dict)
                sentence = ""
                sentence_add = False
        if sentence_add:
            sentence = sentence + char
        if char == ';' and body[i+1] == ' ':
            sentence = ""
            sentence_add = True
    #clear all with 'Q'
    new = []
    for i, s in enumerate(sentences):
        # print(s)
        if 'Q' not in s:
            new.append(s)
    return new

def add_tags(sentences):
    sentence_dict = []
    kkma = Kkma()
    for i, sent in enumerate(sentences):
        sentence_dict.append((sent, kkma.pos(sent)))
    return sentence_dict



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
    '''
    Functions for importing corpus and extracting sentences+tags_dict
    dir = os.getcwd()
    raw = c.open(dir + "/dataset/BGEO0292.txt", "rb", "utf-16")
    data = raw.read() # to be able to manipulate input
    body = extract_body(data)    #extract sentences
    sentences = parse(body)      #extract tags
    sentence_dict = add_tags(sentences) #sentence_dict = [sentence, [(word, tag)]
    '''

    data = "몇 시가 되었는지 경지는 모르고 있었다 ."

    s1 = data.split()
    print(s1)

    # Nivre Parser
    parser = NivreParser(s1)
    parser = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
    print(parser)

    printTree(parser, s1)

if __name__ == '__main__':
    main()
