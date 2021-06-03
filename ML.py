# import tensorflow as tf
# from matplotlib import pyplot as plt
#
# # step 1
# tf.reset_default_graph()
#
# # step 2
# input_data = tf.placeholder(dtype=tf.float32, shape=None)
# output_data = tf.placeholder(dtype=tf.float32, shape=None)
#
# slope = tf.Variable(.5, dtype=tf.float32)
# intercept = tf.Variable(.1, dtype=tf.float32)
# # step 3
#
# model_op = slope * input_data + intercept
#
# error: object = model_op - output_data
# squared_error = tf.square(error)
#
# loss = tf.reduce_mean(squared_error)
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
#
# train = optimizer.minimize(loss)
#
# # start session
# init = tf.global_variables_initializer()
# # user input
# x_vals = [0,  1,  2,  3,  4,  5,  6,    7,  8,    9,    10,   11, 12,   13]
# y_vals = [38, 33, 14, 46, 34, 40, 31.5, 22, 20.5, 27.4, 47.5, 48, 20.5, 20]
#
# with tf.Session() as sess:
#     sess.run(init)
#     for i in range(2000):
#         sess.run(train, feed_dict={input_data: x_vals, output_data: y_vals})
#         if (i % 100 == 0):
#             print(sess.run([slope, intercept]))
#             plt.plot(x_vals, sess.run(model_op, feed_dict={input_data: x_vals}))
#     print(sess.run(loss, feed_dict={input_data: x_vals, output_data: y_vals}))
#     plt.plot(x_vals, y_vals, 'ro', 'Training Data')
#     plt.plot(x_vals, sess.run(model_op, feed_dict={input_data:x_vals, output_data:y_vals}))
#     plt.show()

import tensorflow as tf
import numpy as np


# OS to load files and save checkpoints
import os

image_height = 28
image_width = 28

color_channels = 1

model_name = "mnist"

mnist = tf.contrib.learn.datasets.load_dataset("mnist")

train_data = mnist.train.images
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

eval_data = mnist.test.images
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

category_names = list(map(str, range(10)))

# TODO: Process mnist data

train_data = np.reshape(train_data, (-1, image_height, image_width, color_channels))
eval_data = np.reshape(eval_data, (-1, image_height, image_width, color_channels))

# # Load cifar data from file
#
# image_height = 32
# image_width = 32
#
# color_channels = 3
#
# model_name = "cifar"
#
# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict
#
# cifar_path = './cifar-10-data/'
#
#
# train_data = np.array([])
# train_labels = np.array([])
#
# # Load all the data batches.
# for i in range(1,6):
#     data_batch = unpickle(cifar_path + 'data_batch_' + str(i))
#     train_data = np.append(train_data, data_batch[b'data'])
#     train_labels = np.append(train_labels, data_batch[b'labels'])
#
#
# # Load the eval batch.
# eval_batch = unpickle(cifar_path + 'test_batch')
#
# eval_data = eval_batch[b'data']
# eval_labels = eval_batch[b'labels']
#
# # Load the english category names.
# category_names_bytes = unpickle(cifar_path + 'batches.meta')[b'label_names']
# category_names = list(map(lambda x: x.decode("utf-8"), category_names_bytes))
#
# # TODO: Process Cifar data
#
# def process_data(data):
#     float_data = np.array(data, dtype=float)/255.0
#     reshaped_data = np.reshape(float_data, (-1, color_channels, image_height, image_width))
#     transpose_data = np.transpose(reshaped_data, [0, 2, 3, 1])
#     return transpose_data
# train_data = process_data(train_data)
# eval_data = process_data(eval_data)

class ConvNet:
    def __init__(self, image_height, image_width, channels, num_classes):
        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, image_height, image_width, channels],
                                          name="inputs")
        conv_layer1 = tf.layers.conv2d(self.input_layer, filters=32, kernel_size=[5, 5], activation=tf.nn.relu,padding="same")
        pooling_layer1 = tf.layers.max_pooling2d(conv_layer1, pool_size=[2, 2], strides=2)
        conv_layer2 = tf.layers.conv2d(pooling_layer1, filters=64, kernel_size=[5, 5], padding="same",activation=tf.nn.relu)
        pooling_layer2 = tf.layers.max_pooling2d(conv_layer2, pool_size=[2, 2], strides=2)
        conv_layer3 = tf.layers.conv2d(pooling_layer2, filters=128, kernel_size=[5, 5], padding="same",activation=tf.nn.relu)
        pooling_layer3 = tf.layers.max_pooling2d(conv_layer3, pool_size=[2, 2], strides=2)
        flattened_pooling = tf.layers.flatten(pooling_layer3)
        dense_layer = tf.layers.dense(flattened_pooling, 1024, activation=tf.nn.relu)
        dropout_rate = tf.layers.dropout(dense_layer, rate=.6, training=True)
        outputs = tf.layers.dense(dropout_rate, num_classes)

        self.choice = tf.argmax(outputs, axis=1)
        self.probability = tf.nn.softmax(outputs)
        self.labels = tf.placeholder(dtype=tf.float32, name="labels")
        self.accuracy, self.accuracy_op = tf.metrics.accuracy(self.labels, self.choice)
        one_hot_labels = tf.one_hot(indices=tf.cast(self.labels, dtype=tf.int32), depth=num_classes)
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=outputs)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1)
        self.train_op = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

# TODO: initialize variables and train nn
training_steps = 200
batch_size = 64
path = "./"+model_name+"-cnn/"
load_checkpoint=False
performance_graph = np.array([])

tf.reset_default_graph()
dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
dataset = dataset.shuffle(buffer_size=train_labels.shape[0])
dataset = dataset.batch(batch_size)
dataset = dataset.repeat()
dataset_iterator = dataset.make_initializable_iterator()
next_elem = dataset_iterator.get_next()
cnn = ConvNet(image_height, image_width, color_channels, 10)
saver = tf.train.Saver(max_to_keep=2)

if not os.path.exists(path):
    os.makedirs(path)

# TODO: implement the training loop
with tf.Session() as sess:
    if (load_checkpoint):
        checkpoint = tf.train.get_checkpoint_state(path)
        saver.restore(sess, checkpoint.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(dataset_iterator.initializer)

    for step in range(training_steps):
        current_batch = sess.run(next_elem)
        batch_inputs = current_batch[0]
        batch_labels = current_batch[1]
        sess.run((cnn.train_op, cnn.accuracy_op), feed_dict={cnn.input_layer: batch_inputs, cnn.labels: batch_labels})
        if step % 10 ==0:
            performance_graph = np.append(performance_graph, sess.run(cnn.accuracy))
        if step % 1000 and step > 0:
            current_acc = sess.run(cnn.accuracy)
            print("CURRENT ACCURACY AT STEP " + str(step) + ": " + str(current_acc*100)+"%")
            # print("Saving Checkpoint....")
            # saver.save(sess, path + model_name, step)
    print("Saving final checkpoint for training session.")
    saver.save(sess, path + model_name, step)

import matplotlib.pyplot as plt

plt.plot(performance_graph)
plt.show()

with tf.Session() as sess:
    checkpoint = tf.train.get_checkpoint_state(path)
    saver.restore(sess, checkpoint.model_checkpoint_path)
    #
    # sess.run(tf.local_variables_initializer())
    # for image, label in zip(eval_data, eval_labels):
    #     sess.run(cnn.accuracy_op, feed_dict={cnn.input_layer: [image], cnn.labels: label})
    # print(sess.run(cnn.accuracy*100))
    rows = 4
    cols = 5
    indices = np.random.choice(len(eval_data), rows*cols, replace=False)
    fig, axes = plt.subplots(rows, cols, figsize = (5,5))
    fig.patch.set_facecolor("white")
    image_count = 0
    for idx in indices:
        image_count+=1
        sub = plt.subplot(rows, cols, image_count)
        img = eval_data[idx]
        if(model_name=="mnist"):
            img = img.reshape(28, 28)
        plt.imshow(img)
        guess = sess.run(cnn.choice, feed_dict={cnn.input_layer: [eval_data[idx]]})
        if model_name == "mnist":
            guess_name = str(guess[0])
            actual_name = str(eval_labels[idx])
        else:
            guess_name= category_names[guess[0]].decode('utf-8')
            actual_name=category_names[eval_labels[idx]].decode('utf-8')
        sub.set_title("G: "+guess_name+" A: "+actual_name)
    plt.tight_layout()
    plt.show()
