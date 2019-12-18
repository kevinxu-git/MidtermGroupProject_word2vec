#-*- coding:utf-8 -*-
import os
from lxml import etree
import codecs as c
from io import StringIO, BytesIO
from konlpy.tag import Kkma
from konlpy.utils import pprint


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


def main():
    #load lxml
    dir = os.getcwd()
    raw = c.open(dir + "/dataset/BGEO0292.txt", "rb", "utf-16")
    data = raw.read() # to be able to manipulate input
    body = extract_body(data)
    #extract sentences
    sentences = parse(body)
    #extract tags
    #sentence_dict = [sentence, [(word, tag)]
    sentence_dict = add_tags(sentences)


if __name__ == '__main__':
    main()
