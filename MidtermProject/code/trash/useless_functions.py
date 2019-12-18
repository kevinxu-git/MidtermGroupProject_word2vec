import numpy as np


'''
Input: batch of word couples [[center_word, neighbour1], [cw, n2],..]
Output: One hot vectors of batch items as numpy arrays.
'''
def create_training_vectors(batch, dict_size):
    x_train = [] # input word
    y_train = [] # output word
    for item in batch:
        x_train.append(to_one_hot(item[0], dict_size).transpose())
        y_train.append(to_one_hot(item[1], dict_size).transpose())
        # print(x_train)
        # print(y_train)
        # convert them to numpy arrays
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    assert not np.any(np.isnan(x_train))
    assert not np.any(np.isnan(y_train))
    return x_train.transpose(), y_train.transpose()




def J_ns(c, o, U):
	return -log(U[o, c])



def J_sg(c, U, V, word_index):
	S = 0
	for j in word_index:
		S += J_ns(c, j, U)
	return S


	