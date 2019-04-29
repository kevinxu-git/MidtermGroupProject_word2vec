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

BAD_CHARS = [',', '<', '>', '!', '?', '-','<', ':',';','*','(',')','[',']','`','\'','"','는 ','은 ','과 ','이 ','그','저','가 ','을 ','를 ','에 ','와','나','로','의 ','도 ','께']

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
    sentences = pre_process("박스를 자르던 김인용 씨 가 얼굴에 맺힌 땀을 닦으며 말했다. 그는 올 초부터 이곳에서 캔버스를 만들고 있다. 2016년 문을 연 러블리페이퍼 는 폐지를 줍는 저소득층 노인을 돕는 사회적 기업이다. 인텔은 이익 전망에 대한 실망감을 빌미로 10%에 달하는 폭락을 연출했다. 반면 포드는 1분기 이익과 매출 감소 폭이 시장의 예상보다 작은 것으로 나타나면서 10% 선에서 랠리했다. 국제 유가는 가파르게 하락했다. 뉴욕상업거래소에서 서부텍사스산원유(WTI)가 2.9% 떨어진 배럴당 63.30달러에 거래됐다. 도널드 트럼프 대통령이 석유수출국기구(OPEC) 정책자와 전화 통화를 갖고 유가 안정을 위해 대응할 것을 요구했다고 밝히면서 ‘팔자’가 쏟아졌다. E트레이드 파이낸셜은 투자 보고서를 내고 “GDP 성장률이 크게 상승했지만 비즈니스 사이클이 하강 기류로 접어들었고, 기업 이익 전망이 만족스럽지 못하다”고 평가했다. 펜 뮤추얼 애셋 매니지먼트의 지웨이 렌 포트폴리오 매니저는 마켓워치와 인터뷰에서 “인텔의 이익 전망이 IT 섹터에 부담을 가했다”고 말했다. [뉴욕=뉴스핌] 황숙혜 특파원 = 도널드 트럼프 미국 대통령이 석유수출국기구(OPEC)에 공급 확대를 주문한 데 따라 국제 유가가 큰 폭으로 떨어졌다. 서부텍사스산원유(WTI) 일간 추이 [출처=인베스팅닷컴 앞서 사우디 아라비아가 미국의 이란 제재 면제 종료에 즉각 대응할 필요가 없다는 입장을 제시한 데 대해 반기를 든 셈이다. 26일(현지시각) 뉴욕상업거래소에서 서부텍사스산원유(WTI)가 1.91달러(2.9%) 급락하며 배럴당 63.30달러에 거래됐다. 장중 한 때 WTI는 4% 후퇴하며 배럴당 62달러 선으로 밀린 뒤 낙폭을 축소했다. 국제 벤치마크인 브렌트유 역시 장중 한 때 4% 가량 내리 꽂히며 배럴당 71달러 선으로 밀리며 미국의 이란 강경책 발표 이후 상승분 가운데 상당 부분을 토해냈다.")
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
    font_location = 'Typo_DodamM.ttf'
    # ex - 'C:/asiahead4.ttf'
    prop = fm.FontProperties(fname = font_location)
    # print(font_name)
    # pyplot.rc('font', family = prop)
    # pyplot.rc('font', **{'sans-serif' : 'Arial',
    #                      'family' : 'sans-serif'})

    # word2vec
    U, V = word2vec_skip_gram(dictionary, sentences)

    # Save U and V
    # f = open("output.txt", "w")
    # f.write("U = \n")
    # f.write(str(U))
    # f.write("\nV = \n")
    # f.write(str(V))
    # f.close()
    np.savetxt('matrix_U.txt', U, fmt='%f')
    np.savetxt('matrix_V.txt', V, fmt='%f')

    print(U)
    # print(U[1,1])
    # b = np.loadtxt('test1.txt')
    # print(b[1,1])
    # print(U == b)

    # PCA
    X = U+V
    pca = PCA(n_components = 2)
    result = pca.fit_transform(X.transpose())
    # print("Explained variance ratio : " ,pca.explained_variance_ratio_, end = "\n")
    # print("Singular values : ",pca.singular_values_, end = "\n")

    # Plot of word vectors
    pyplot.scatter(result[:, 0], result[:, 1])

    for i in range(len(U)):
            pyplot.annotate(int2word[i], xy=(result[i, 0], result[i, 1]), fontproperties = prop)
    pyplot.xlabel(u'언녕', fontproperties = prop)
    pyplot.show()

if __name__ == '__main__':
    main()
