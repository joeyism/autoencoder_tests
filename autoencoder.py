import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
h, w = (28, 28)


def create_model(h, w, z, learning_rate = 0.01):
    size = h*w*z
    # Input
    X = tf.placeholder(tf.float32, [None, size])


    # Encoder
    W1 = tf.Variable(tf.random_normal([size, 256]))
    h1 = tf.Variable(tf.random_normal([256]))
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(X, W1), h1))


    W2 = tf.Variable(tf.random_normal([256, 128]))
    h2 = tf.Variable(tf.random_normal([128]))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, W2), h2))


    # Decoder
    W3 = tf.Variable(tf.random_normal([128, 256]))
    h3 = tf.Variable(tf.random_normal([256]))
    layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, W3), h3))

    W4 = tf.Variable(tf.random_normal([256, size]))
    h4 = tf.Variable(tf.random_normal([size]))
    layer4 = tf.nn.sigmoid(tf.add(tf.matmul(layer3, W4), h4))

    y_pred = layer4
    y_true = X
    loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    init = tf.global_variables_initializer()

    def autoencode_and_generate(input_images, images_to_run, no_of_epochs=300):
        with tf.Session() as sess:
            sess.run(init)

            for i in tqdm(range(1, no_of_epochs), desc="Training"):
                _, loss = sess.run([optimizer, loss], feed_dict={X:input_images})

                if i % 100 == 0:
                    print("Step {}: Loss: {}".format(i, loss))

        return sess.run(y_pred, feed_dict={X:images_to_run})
    return autoencode_and_generate

