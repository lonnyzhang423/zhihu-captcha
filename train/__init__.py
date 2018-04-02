import os
import logging

import numpy as np
import tensorflow as tf
from config import *
from utils import *

__all__ = ["predict_captcha"]

graph = tf.Graph()
with graph.as_default():
    normal_sess = tf.Session()
    latest_checkpoint = tf.train.latest_checkpoint(checkpoints_dir())
    if latest_checkpoint:
        head, tail = os.path.split(latest_checkpoint)
        tf.train.import_meta_graph(os.path.join(checkpoints_dir(), tail + ".meta"))
        tf.train.Saver().restore(normal_sess, latest_checkpoint)

        X = graph.get_tensor_by_name("input/input_x:0")
        Y = graph.get_tensor_by_name("input/input_y:0")
        keep_prob = graph.get_tensor_by_name("keep_prob/keep_prob:0")
        logits = graph.get_tensor_by_name("final_output/logits:0")
        predict = tf.argmax(tf.reshape(logits, [-1, CAPTCHA_LEN, CHAR_SET_LEN]), 2)


# noinspection PyBroadException
def predict_captcha(image):
    """
    :param image: image base64
    :return: train captcha code
    """
    try:
        image = img2array(image)
        max_idx = normal_sess.run(predict, feed_dict={X: [image], keep_prob: 1.0})

        char_idx = max_idx[0].tolist()
        vector = np.zeros(CAPTCHA_LEN * CHAR_SET_LEN)
        i = 0
        for idx in char_idx:
            vector[i * CHAR_SET_LEN + idx] = 1
            i += 1
        return vector2text(vector)
    except BaseException:
        logging.warning("Predict captcha exception!", exc_info=True)
        return INVALID_CAPTCHA


def test_accuracy():
    total = hits = 0
    for i in range(100):
        total += 1
        text, image = next_test_text_and_image()
        predict_text = predict_captcha(image)
        hit = False
        if text == predict_text:
            hit = True
            hits += 1
        print("Correct:", text, "Predict:", predict_text, "hit:", hit)

    print("Evaluate {} times. Accuracy:{}".format(total, hits / 100))


if __name__ == '__main__':
    test_accuracy()
