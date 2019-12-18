import tensorflow as tf
import numpy as np

EMBEDDING_SIZE = 10
LEARNING_RATE = 3
EPOCHS = 100

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

def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros((vocab_size, 1))
    temp[data_point_index] = 1
    return temp



def main():
    #import data
    '''
    Could not import good data in time
    input_data = [current_state]
    current_state = [step, stack, buffer]
    Words in stack & buffer are represented as one_hot vectors where one position is a
    unique [word, label] from the dataset.
    Output_data = the 4 the transition actions [left, right, reduce, shift]
    '''

    int2word, word2int = create_dictionary(sentences)
    batch = generate_batch(sentences)

    #placeholder
    x = tf.placeholder(tf.float32, shape=(None, dict_size))
    y_label = tf.placeholder(tf.float32, shape=(None, dict_size))


    #create model
    W1 = tf.Variable(tf.random_normal([dict_size, EMBEDDING_SIZE]))
    b1 = tf.Variable(tf.random_normal([EMBEDDING_SIZE])) #bias
    hidden_representation = tf.add(tf.matmul(x,W1), b1)

    W2 = tf.Variable(tf.random_normal([EMBEDDING_SIZE, dict_size]))
    b2 = tf.Variable(tf.random_normal([dict_size]))
    prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))

    sess = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)

    #Loss function
    loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))

    #Step function
    step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

    #train
    for _ in range(EPOCHS):
        session.run(step, feed_dict={x: x_train, y_label: y_train})
        print('loss is : ', session.run(loss, feed_dict={x: x_train, y_label: y_train}))


    #save
    saver = tf.train.Saver()
    save_path = saver.save(sess, SAVE_PATH + "/tmp/oracle.ckpt")
    print("Model saved in path: %s" % save_path)

if __name__ == '__main__':
    main()
