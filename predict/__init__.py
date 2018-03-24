import os
import logging

import numpy as np
import tensorflow as tf
from config import *
from utils import *

__all__ = ["predict_captcha"]

normal_graph = tf.Graph()
with normal_graph.as_default():
    normal_sess = tf.Session()
    latest_checkpoint = tf.train.latest_checkpoint(predict_normal_checkpoints_dir())
    if latest_checkpoint:
        head, tail = os.path.split(latest_checkpoint)
        tf.train.import_meta_graph(os.path.join(predict_normal_checkpoints_dir(), tail + ".meta"))
        tf.train.Saver().restore(normal_sess, latest_checkpoint)

        X_normal = normal_graph.get_tensor_by_name("input/input_x:0")
        Y_normal = normal_graph.get_tensor_by_name("input/input_y:0")
        keep_prob_normal = normal_graph.get_tensor_by_name("keep_prob/keep_prob:0")
        logits_normal = normal_graph.get_tensor_by_name("final_output/logits:0")
        predict_normal = tf.argmax(tf.reshape(logits_normal, [-1, CAPTCHA_LEN, CHAR_SET_LEN]), 2)

bold_graph = tf.Graph()
with bold_graph.as_default():
    bold_sess = tf.Session()
    latest_checkpoint = tf.train.latest_checkpoint(predict_bold_checkpoints_dir())
    if latest_checkpoint:
        head, tail = os.path.split(latest_checkpoint)
        tf.train.import_meta_graph(os.path.join(predict_bold_checkpoints_dir(), tail + ".meta"))
        tf.train.Saver().restore(bold_sess, latest_checkpoint)

        X_bold = bold_graph.get_tensor_by_name("input/input_x:0")
        Y_bold = bold_graph.get_tensor_by_name("input/input_y:0")
        keep_prob_bold = bold_graph.get_tensor_by_name("keep_prob/keep_prob:0")
        logits_bold = bold_graph.get_tensor_by_name("final_output/logits:0")
        predict_bold = tf.argmax(tf.reshape(logits_bold, [-1, CAPTCHA_LEN, CHAR_SET_LEN]), 2)


# noinspection PyBroadException
def predict_captcha(image, clazz=0):
    """
    :param image: image base64
    :param clazz: 0: normal captcha
                  1: bold captcha
                  -1: invalid captcha
    :return: predict captcha code
    """
    try:
        if clazz != 0 and clazz != 1:
            return INVALID_CAPTCHA
        image = img2array(image)
        if clazz == 0:
            max_idx = normal_sess.run(predict_normal, feed_dict={X_normal: [image], keep_prob_normal: 1.0})
        else:
            max_idx = bold_sess.run(predict_bold, feed_dict={X_bold: [image], keep_prob_bold: 1.0})

        char_idx = max_idx[0].tolist()
        vector = np.zeros(CAPTCHA_LEN * CHAR_SET_LEN)
        i = 0
        for idx in char_idx:
            vector[i * CHAR_SET_LEN + idx] = 1
            i += 1
        return vector2text(vector)
    except BaseException:
        logging.warning("Predict captcha exception! Class:%s", clazz, exc_info=True)
        return INVALID_CAPTCHA


def test_accuracy():
    hits = 0
    for i in range(100):
        text, image = next_normal_text_and_image()
        predict_text = predict_captcha(image)
        hit = False
        if text == predict_text:
            hit = True
            hits += 1
        print("Correct:", text, "Predict:", predict_text, "hit:", hit)

    print("Test 100 captchas. Accuracy:", hits / 100)


if __name__ == '__main__':
    test_accuracy()
