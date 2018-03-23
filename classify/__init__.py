import os
import logging

import tensorflow as tf

from config import *
from utils import *

__all__ = ["classify_captcha"]

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    latest_checkpoint = tf.train.latest_checkpoint(classify_checkpoints_dir())
    if latest_checkpoint:
        head, tail = os.path.split(latest_checkpoint)
        tf.train.import_meta_graph(os.path.join(classify_checkpoints_dir(), tail + ".meta"))
        tf.train.Saver().restore(sess, latest_checkpoint)

        X = graph.get_tensor_by_name("input/input_x:0")
        Y = graph.get_tensor_by_name("input/input_y:0")
        keep_prob = graph.get_tensor_by_name("keep_prob/keep_prob:0")
        logits = graph.get_tensor_by_name("final_output/logits:0")
        predict = tf.argmax(tf.reshape(logits, [-1, NUM_CLASSIFY_CLASSES]), 1)


# noinspection PyBroadException
def classify_captcha(image):
    """
        classify captcha image
    Return:
        0: normal captcha
        1: bold captcha
        -1: invalid image
    """
    try:
        image = img2array(image)
        max_idx = sess.run(predict, feed_dict={X: [image], keep_prob: 1.0})
        clazz = max_idx[0].tolist()
        return clazz
    except BaseException:
        logging.warning("Classify captcha exception!", exc_info=True)
        return -1


def test_accuracy():
    with open("../samples/bold_captcha_base64.txt", "r") as f:
        hits = 0
        for line in f:
            code, image = line.split(":")
            hits += classify_captcha(image)
        print(hits)


if __name__ == '__main__':
    test_accuracy()
