#-*- coding:utf-8 -*-
import os
from lxml import etree
import codecs as c
from io import StringIO, BytesIO
from konlpy.tag import Hannanum
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
        # if char == '/':
        #     next = body[i]
        #     next_int = 0
        #     tag = ""
        #     while next != ' ' and next != '(' and next != ')':
        #         tag = tag + next
        #         next = body[i+next_int]
        #         next_int += 1
        #     next = body[i]
        #     next_int = i
        #     word = ""
        #     while body[next_int] != ' ':
        #         next_int -= 1
        #     word = body[i-next_int:i]
        #     dictionary.append((word, tag))

    #clear all with 'Q'
    new = []
    for i, s in enumerate(sentences):
        # print(s)
        if 'Q' not in s:
            new.append(s)
    return new

def extract_sentences(data):
    body = extract_body(data)
    sentences = parse(body)

    kkma = Hannanum()
    print(kkma.pos(sentences[1]))
    # print(sentences)

def main():
    #load lxml
    dir = os.getcwd()
    raw = c.open(dir + "/dataset/BGEO0292.txt", "rb", "utf-16")
    data = raw.read() # to be able to manipulate input
    # print(data)
    #extract sentences
    data = extract_sentences(data)
    #extract tags
    #[sentences, [(word, tag)]


if __name__ == '__main__':
    main()
