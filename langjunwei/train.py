import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

tf.set_random_seed(2018)
# reset the graph
tf.reset_default_graph()
# define hyper-parameters
batch_size = 1290
learning_rate = 0.001
keep_prob_hp = 0.3
output_depths = [90, 45, 20]
save_path = 'models/904520_1290_batch/model.ckpt'


def hidden_layer(layer_input, output_depth, scope='hidden_layer', reuse=None):
    input_depth = layer_input.get_shape()[-1]
    print('input_depth = ', input_depth)
    with tf.variable_scope(scope, reuse=reuse):
        # use truncated_normal_initializer to make distribution more concentrated
        w = tf.get_variable(name='weight',
                            shape=(input_depth, output_depth),
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable(name='bias',
                            shape=(output_depth),
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.1))
        net = tf.matmul(layer_input, w) + b
        # drop out
        net = tf.nn.dropout(net, keep_prob)
        return net


def DNN(input, output_depths, scope='DNN', reuse=None):
    net = input
    for i, output_depth in enumerate(output_depths):
        net = hidden_layer(layer_input=net,
                           output_depth=output_depth,
                           scope='layer{}'.format(i),
                           reuse=reuse)
        net = tf.nn.relu(net)

    # the last layer should be 10 dimensions to classify, don't drop out
    net = hidden_layer(layer_input=net,
                       output_depth=8,
                       scope='classification',
                       reuse=reuse)
    return net


# define the placeholder rather than variable for changing input_data during training and testing
input_ph = tf.placeholder(dtype=tf.float32,
                          shape=(None, 102))
label_ph = tf.placeholder(dtype=tf.float32,
                          shape=(None, 8))
keep_prob = tf.placeholder(dtype=tf.float32)
# create a 4-layer neural networks whose hidden sizes are [60, 30, 15, 8]
dnn = DNN(input=input_ph,
          output_depths=output_depths)

with tf.name_scope('MSE_loss'):
    loss = tf.losses.mean_squared_error(labels=label_ph, predictions=dnn)
    loss_mean = tf.losses.mean_squared_error(labels=label_ph, predictions=dnn, reduction=tf.losses.Reduction.MEAN)
    tf.summary.scalar('MSE_loss', loss)

with tf.name_scope('train'):
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss=loss)

# get data for training and testing
data_input = pd.read_csv('data/train_input.csv')
data_output = pd.read_csv('data/train_output.csv')
input_numpy = np.array(data_input['ratio'])  # shape=(66453, ) should be reshape to (n, 51, )
input_numpy = input_numpy.reshape(-1, 51)  # (index_of_samples, ratios)
cut_in_loss_numpy = np.array(data_input['cut_in_loss'])  # shape=(66453, ) should be reshape to (n, 51, )
cut_in_loss_numpy = cut_in_loss_numpy.reshape(-1, 51)  # (index_of_samples, cut_in_losses)
input_numpy = np.hstack((input_numpy, cut_in_loss_numpy))  # shape=(n, 102), input
label_numpy = np.array(data_output[['y11', 'x12', 'x13', 'y14', 'y21', 'x22', 'x23', 'y24']])  # shape=(n, 8), output

test_data_input = pd.read_csv('data/test_input.csv')
test_data_output = pd.read_csv('data/test_output.csv')
test_input_numpy = np.array(test_data_input['ratio'])  # shape=(66453, ) should be reshape to (n, 51, )
test_input_numpy = test_input_numpy.reshape(-1, 51)  # (index_of_samples, ratios)
test_cut_in_loss_numpy = np.array(test_data_input['cut_in_loss'])  # shape=(66453, ) should be reshape to (n, 51, )
test_cut_in_loss_numpy = test_cut_in_loss_numpy.reshape(-1, 51)  # (index_of_samples, cut_in_losses)
test_input_numpy = np.hstack((test_input_numpy, test_cut_in_loss_numpy))  # shape=(n, 102), input
test_label_numpy = np.array(test_data_output[['y11', 'x12', 'x13', 'y14', 'y21', 'x22', 'x23', 'y24']])  # shape=(n, 8), output

# normalize the input data(very important or NaN)
# scaler = preprocessing.StandardScaler().fit(input_numpy)
# input_numpy = scaler.transform(input_numpy)
# scaler = preprocessing.StandardScaler().fit(test_input_numpy)
# test_input_numpy = scaler.transform(test_input_numpy)
# scaler = preprocessing.StandardScaler().fit(label_numpy)
# label_numpy = scaler.transform(label_numpy)
# scaler = preprocessing.StandardScaler().fit(test_label_numpy)
# test_label_numpy = scaler.transform(test_label_numpy)

# start to train
merged = tf.summary.merge_all()
sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=1)
# open file writer
# train_writer = tf.summary.FileWriter('data/train_log', sess.graph)
# test_writer = tf.summary.FileWriter('data/test_log', sess.graph)

sess.run(tf.global_variables_initializer())

for e in range(100000):
    # shuffle the dataset
    permutation = np.random.permutation(input_numpy.shape[0])
    shuffled_input_numpy = input_numpy[permutation, :]
    shuffled_label_numpy = label_numpy[permutation, :]
    for i in range(input_numpy.shape[0] // batch_size):
        images = input_numpy[batch_size * i: batch_size * (i + 1), :]
        labels = label_numpy[batch_size * i: batch_size * (i + 1), :]
        sess.run(train_op, feed_dict={input_ph: images, label_ph: labels, keep_prob: keep_prob_hp})
    if (e + 1) % 1000 == 0:
        images = input_numpy[0: 3896, :]
        labels = label_numpy[0: 3896, :]
        test_images = test_input_numpy
        test_labels = test_label_numpy
        # calculate the loss of train_set and test_set
        sum_train, loss_train = sess.run([merged, loss], feed_dict={input_ph: images, label_ph: labels, keep_prob: 1})
        # train_writer.add_summary(sum_train, e)
        sum_test, loss_test = sess.run([merged, loss], feed_dict={input_ph: test_images, label_ph: test_labels, keep_prob: 1})
        # test_writer.add_summary(sum_test, e)
        print('STEP {}: train_loss: {:.6f} test_loss: {:.6f}'.format(
            e + 1, loss_train, loss_test))
        # save the model
        # saver.save(sess, save_path, global_step=e+1)
        print('model saving done...')
        # print('loss_mean: \n', sess.run(loss_mean, feed_dict={input_ph: test_images, label_ph: test_labels, keep_prob: 1}))
        print('labels: \n', test_labels[0: 2, :])
        print('predictions: \n', sess.run(dnn, feed_dict={input_ph: test_images[0: 2, :], keep_prob: 1}))

# close file writer
# train_writer.close()
# test_writer.close()
print('Training is done')
print('-' * 30)

# calculate the loss of all the train_set
train_loss = []

for e in range(input_numpy.shape[0] // batch_size):
    # image, label = train_set.next_batch(100)
    image = input_numpy[batch_size * e: batch_size * (e + 1), :]
    label = label_numpy[batch_size * e: batch_size * (e + 1), :]
    loss_train = sess.run(loss, feed_dict={input_ph: image, label_ph: label, keep_prob: 1})
    train_loss.append(loss_train)

print('Train loss: {:.6f}'.format(np.array(train_loss).mean()))

# calculate the loss of all the test_set
test_loss = []

for e in range(test_input_numpy.shape[0] // 100):
    image = test_input_numpy
    label = test_label_numpy
    loss_test= sess.run(loss, feed_dict={input_ph: image, label_ph: label, keep_prob: 1})
    test_loss.append(loss_test)

print('Test loss: {:.6f}'.format(np.array(test_loss).mean()))

sess.close()