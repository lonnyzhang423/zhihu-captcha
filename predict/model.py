import os
import tensorflow as tf

from config import *
from utils import *

tf.reset_default_graph()
with tf.name_scope("input"):
    X = tf.placeholder(tf.float32, [None, IMG_WIDTH * IMG_HEIGHT], name="input_x")
    Y = tf.placeholder(tf.float32, [None, CAPTCHA_LEN * CHAR_SET_LEN], name="input_y")
    x_img = tf.reshape(X, [-1, IMG_HEIGHT, IMG_WIDTH, 1], name="reshaped_input_x")

with tf.name_scope("keep_prob"):
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")


def random_value(shape, alpha=0.01):
    return alpha * tf.random_normal(shape=shape)


def inference():
    """
        Build model
    Return:
        output tensor with the computed logits, float, [batch_size,n_classes]
    """

    # 3 convolutional layers
    with tf.variable_scope("conv1"):
        weights = tf.get_variable("weights", initializer=random_value(shape=[3, 3, 1, 32]))
        biases = tf.get_variable("biases", initializer=random_value(shape=[32], alpha=0.1))
        conv = tf.nn.conv2d(x_img, weights, strides=[1, 1, 1, 1, ], padding="SAME")
        conv1 = tf.nn.relu(tf.nn.bias_add(conv, biases))
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        pool1 = tf.nn.dropout(pool1, keep_prob=keep_prob)

    with tf.variable_scope("conv2"):
        weights = tf.get_variable("weights", initializer=random_value(shape=[3, 3, 32, 64]))
        biases = tf.get_variable("biases", initializer=random_value(shape=[64], alpha=0.1))
        conv = tf.nn.conv2d(pool1, weights, strides=[1, 1, 1, 1], padding="SAME")
        conv2 = tf.nn.relu(tf.nn.bias_add(conv, biases))
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        pool2 = tf.nn.dropout(pool2, keep_prob=keep_prob)

    with tf.variable_scope("conv3"):
        weights = tf.get_variable("weights", initializer=random_value(shape=[3, 3, 64, 64]))
        biases = tf.get_variable("biases", initializer=random_value(shape=[64], alpha=0.1))
        conv = tf.nn.conv2d(pool2, weights, strides=[1, 1, 1, 1], padding="SAME")
        conv3 = tf.nn.relu(tf.nn.bias_add(conv, biases))
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        pool3 = tf.nn.dropout(pool3, keep_prob=keep_prob)

    # Fully connected layer
    with tf.variable_scope("fc"):
        weights = tf.get_variable("weights", initializer=random_value(shape=[19 * 8 * 64, 1024]))
        biases = tf.get_variable("biases", initializer=random_value(shape=[1024], alpha=0.1))

        dense = tf.reshape(pool3, [-1, weights.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, weights), biases))
        dense = tf.nn.dropout(dense, keep_prob=keep_prob)

    with tf.variable_scope("final_output"):
        weights = tf.get_variable("weights",
                                  initializer=random_value(shape=[1024, CAPTCHA_LEN * CHAR_SET_LEN]))
        biases = tf.get_variable("biases",
                                 initializer=random_value(shape=[CAPTCHA_LEN * CHAR_SET_LEN], alpha=0.1))
        output = tf.add(tf.matmul(dense, weights), biases, name="logits")

    return output


def losses(logits, labels):
    """
    Args:
        logits: logits tensor,[batcha_size,n_classes]
        labels: label tensor,[batch_size]
    Return:
        loss tensor of float type
    """
    with tf.variable_scope("loss") as scope:
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy, name=scope.name)
    tf.summary.scalar("loss", loss)
    return loss


def train_step(loss, learning_rate=0.001):
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    return optimizer


def evaluation(logits, labels):
    with tf.variable_scope("accuracy"):
        predict = tf.reshape(logits, [-1, CAPTCHA_LEN, CHAR_SET_LEN])
        correct = tf.reshape(labels, [-1, CAPTCHA_LEN, CHAR_SET_LEN])

        max_idx_p = tf.argmax(predict, 2)
        max_idx_l = tf.argmax(correct, 2)
        correct_prediction = tf.equal(max_idx_p, max_idx_l)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar("accuracy", accuracy)
    return accuracy


def feed_dict(training=True, clazz=0):
    x, y = next_predict_batch(clazz=clazz)
    if training:
        return {X: x, Y: y, keep_prob: 0.7}
    else:
        return {X: x, Y: y, keep_prob: 1.0}


def checkpoints_dir(clazz=0):
    if clazz == 0:
        return predict_normal_checkpoints_dir()
    else:
        return predict_bold_checkpoints_dir()


def start_train(clazz=0):
    logits = inference()
    loss = losses(logits, Y)
    train_op = train_step(loss)
    accuracy = evaluation(logits, Y)

    saver = tf.train.Saver(max_to_keep=2)
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        if clazz == 0:
            summary = tf.summary.FileWriter("normal_logs", sess.graph)
        else:
            summary = tf.summary.FileWriter("bold_logs", sess.graph)

        sess.run(tf.global_variables_initializer())
        try:
            for step in range(0, 10000):
                _, loss_ = sess.run([train_op, loss], feed_dict=feed_dict(True, clazz=clazz))
                print("Step:", step, "Loss:", loss_)

                if step % 100 == 0:
                    logs, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False, clazz=clazz))
                    summary.add_summary(logs, step)
                    print("Step:", step, "Accuracy:", acc)

                if step and step % 1000 == 0:
                    file = os.path.join(checkpoints_dir(clazz=clazz), "predict_model")
                    saver.save(sess, file, global_step=step)
        except KeyboardInterrupt as e:
            file = os.path.join(checkpoints_dir(clazz=clazz), "predict_model")
            saver.save(sess, file, global_step=100000)
            raise e


if __name__ == '__main__':
    start_train(clazz=1)
