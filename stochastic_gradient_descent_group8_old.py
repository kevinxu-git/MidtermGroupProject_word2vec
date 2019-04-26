# Korean corpus
from konlpy.corpus import kolaw # imports the textfile pointer
from konlpy import utils

import matplotlib.pyplot as pyplot
import random as r
import numpy as np
from math import *

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score

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
	vocab_size = len(U)
	y_hat = np.zeros((vocab_size, 1))
	S = 0
	for w in range(vocab_size):
		print(U[:, w].transpose().dot(V[:, c]))
		S += exp(U[:, w].transpose().dot(V[:, c]))
	for o in range(vocab_size):
		y_hat[o] = exp(U[:, o].transpose().dot(V[:, c]))

	return y_hat/S

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
##        		print(J_ns_deriv_u(c, j, U, V, w, y, y_h))
        # if the next center word is not the center word in the batck we are working on :
        # then we stop. return the vector grad and the indice of the next mini batch
        if i+1 <= len(batch) - 1 :
            if c != batch[i+1][0]:
            	# print(i)
            	return grad, i+1
    return grad, i+1

def main():
    # Import data, clean up and structure
    sentences = pre_process("L’extrait et le message ont été ciselés avec soin pour les réseaux sociaux et ils ont fait mouche : les partages se comptent par milliers, contribuant à donner de l’ampleur à la désinformation sur le pacte de Marrakech. Les vidéos ont, quant à elles, été visionnées plus de 50 000 fois sur les différentes plates-formes. D’autres sites Internet l’ont également relayée à leur tour. Ces scores n’ont rien d’anormal pour @fandetv, un compte Twitter influent au sein de la droite et de l’extrême droite en ligne. Il est notamment suivi par Marion Maréchal, Fabrice Robert, chef de file des identitaires, ou encore par le compte officiel de la Manif pour tous.Il faut dire que le compte ne ménage pas sa peine, avec plus de 130 000 messages postés depuis 2009. Parmi ses publications qui ont fait sensation, on trouve une intervention télévisée du maire RN de Beaucaire, Julien Sanchez, qualifiant une porte-parole écologiste de « baba cool » qui « n’y conna[ît] rien » au sujet de l’immigration. Un autre extrait à succès, visionné près de 200 000 fois sur la chaîne YouTube The Boss (avant que celle-ci soit mise hors ligne), montre le polémiste Gilles-William Goldnadel interpeller Jean-Michel Aphatie, le qualifiant de « caricature de journaliste ».Six comptes Twitter, 100 000 abonnés et plusieurs chaînes YouTube fandetv se montre par ailleurs assez peu scrupuleux dans ses choix. En février 2018, il a largement mis en avant des propos mensongers de Jean Messiha, un membre du bureau national du RN. Ce dernier assurait sur CNews que l’aide médicale d’Etat (AME) était accordée sans condition aux étrangers en situation irrégulière – ce qui est faux, car il y a bien des conditions de ressources et de résidence stable et régulière. Qu’importe : l’intervention de M. Messiha a été visionnée près de 20 000 fois par la seule grâce d’un tweet de fandetv.Ces scores sont loin d’être anodins lorsqu’on les compare aux audiences de ses émissions, qui se chiffrent généralement en dizaines de milliers de téléspectateurs. « Je suis un simple anonyme qui partage des vidéos de temps en temps sur Twitter et souhaiterait rester dans l’anonymat », a pourtant relativisé l’auteur du compte lorsque nous avons cherché à le contacter par mail, refusant de s’entretenir avec nous.Son activisme en ligne va pourtant bien au-delà de ce seul compte Twitter. Selon les informations que nous avons réunies, c’est un certain Cédric D. qui se cache derrière @fandetv. Cet homme n’anime en réalité pas seulement un, mais au moins six comptes sur le réseau social, qui totalisent plus de 100 000 abonnés, ainsi que plusieurs chaînes YouTube qui représentent des dizaines de millions de vues. Et ses activités vont bien au-delà, avec de multiples sites Internet et des activités commerciales en ligne qui semblent s’inscrire hors du cadre prévu par la loi. Un curieux profil, entre activisme en ligne et affaires douteuses.")
    print("Sentences : ")
    print(sentences)

    # Korean corpus
    # data = kolaw.open('constitution.txt').read()
    # sentences = pre_process(data)

    int2word, word2int = create_dictionary(sentences[0: ])
    dictionary = create_dictionary(sentences[0:500])
    print("\nDictionary : ")
    print(dictionary)
    print(list(dictionary[1]))
    print('is' in dictionary[1])


    #Generate batch
    batch = generate_batch(WINDOW_SIZE, sentences, word2int)
    print("\nbatch")
    print(batch)
    print(len(batch))

##    # Create one hot vectors
    dict_size = len(int2word)
##    x_train, y_train = create_training_vectors(batch, dict_size)
##
##    print("\nnb of words : ")
##    print(len(int2word))
##
##    print("\nV = ")
##    print(x_train)
##    print("\n y true empirical distribution = ")
##    print(y_train)
##
####    # the context windows for center word c
##    c = 1
##    context_windows = y_train[:, 1:3]
##    # print(np.where(context_windows[:,1] == 1)[0][0])
##
##    # The words index in the context windows
##    words_index = []
##    for i in range(len(context_windows[0])):
##    	words_index.append(np.where(context_windows[:,i] == 1)[0][0])
##    print(words_index)
##
##    print(J_sg(c, np.eye(6, 6)+2, np.eye(6, 6), words_index))

    voc_size = dict_size
    print("\nsize dico = ")
    print(voc_size)



    # SDG
    U = np.zeros((voc_size, voc_size))
    V = np.zeros((voc_size, voc_size))

    for i in range(voc_size):
        for j in range(voc_size):
            U[i,j]=r.uniform(0,1)
            V[i,j]=r.uniform(0,1)
    
    print("\nU = ")
    print(U)

##    print(y_hat(1, U, V))
##
##    print("\nder / v = ")
##    print(J_ns_deriv_v(1, 2, np.eye(6, 6), to_one_hot(1, 6), y_hat(1, U, V)))
##    print(J_ns_deriv_u(1, 2, U, V, 2, to_one_hot(1, 6), y_hat(1, U, V)))
    #print(J_ns_deriv_u(0, 2, np.eye(6, 6), y_train[:, 1], x_train, 0, y_hat(1, U, V)))


    theta = np.vstack((V.transpose(), U.transpose()))
    print("\ntheta ini = ")
    print(theta)
    alpha = 0.1
    # while True:
    i = 0
    # for z in range(len(batch)):
    while i != len(batch):
##    	print("\n\n\n\nCenter word = ")
##    	print(batch[i][0])
    	theta_grad, i = grad_J(batch[i][0], theta, batch, i)
    	theta = theta - alpha * theta_grad
##
##    print("\ntheta grad = ")
##    print(theta)
##    print("\ni=")
##    print(i)


    U, V = U_V_from_theta(theta)

    print(U)
    print(V)
    print(len(U))

    A=U+V

    X = A[:, 1:len(U)-1]  
    y = A[0,:]  

    print(X)
    print(y)

    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    
    pyplot.scatter(result[:, 0], result[:, 1])

    words = list(dictionary[1])
    for i in range(len(U)):
            pyplot.annotate(int2word[i], xy=(result[i, 0], result[i, 1]))
    pyplot.show()

    

    print("\n explained variance ratio : " ,pca.explained_variance_ratio_)  

    print("\n singular values : ",pca.singular_values_)  

##
##    # Predicting the Test set results
##    y_pred = classifier.predict(X_test)
##
##    cm = confusion_matrix(y_test, y_pred)  
##    print(cm)  
##    print('Accuracy' + accuracy_score(y_test, y_pred))  
    
##    print("\n\n\nU = ")
##    print(U)
##    print("\n\n\nV = ")
##    print(V)
##
##    print(U+V)
##    A= U+V
##
##    distance = 0
##    for i in range(voc_size):
##        distance += sqrt((A[i,0] - A[i,5])**2)

##    print(distance)

if __name__ == '__main__':
    main()
