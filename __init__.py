import os
import io
import base64

import numpy as np
import tensorflow as tf

from PIL import Image
from data import *
from model import check_points_dir

__all__ = ["predict_captcha"]

tf.reset_default_graph()
sess = tf.Session()
tf.train.import_meta_graph(os.path.join(check_points_dir(), "captcha_model-44000.meta"))
tf.train.Saver().restore(sess, tf.train.latest_checkpoint(check_points_dir()))
graph = tf.get_default_graph()

X = graph.get_tensor_by_name("input/input_x:0")
Y = graph.get_tensor_by_name("input/input_y:0")
keep_prob = graph.get_tensor_by_name("keep_prob/keep_prob:0")
logits = graph.get_tensor_by_name("final_output/logits:0")
predict = tf.argmax(tf.reshape(logits, [-1, CAPTCHA_LEN, CHAR_SET_LEN]), 2)


def _flatten_image(image):
    if isinstance(image, str):
        image = base64.decodebytes(image.encode("utf8"))
        image = Image.open(io.BytesIO(image)).convert("L")
        image = np.array(image).flatten() / 255
    return image


def predict_captcha(image):
    image = _flatten_image(image)
    max_idx = sess.run(predict, feed_dict={X: [image], keep_prob: 1.0})
    char_idx = max_idx[0].tolist()
    vector = np.zeros(CAPTCHA_LEN * CHAR_SET_LEN)
    i = 0
    for idx in char_idx:
        vector[i * CHAR_SET_LEN + idx] = 1
        i += 1
    return vector2text(vector)


if __name__ == '__main__':
    with open("hits_captcha_base64.txt", "r") as f:
        hits = total = 0
        for line in f:
            total += 1
            correct_text, image = line.split(":")
            predict_text = predict_captcha(image)
            hit = False
            if correct_text == predict_text:
                hits += 1
                hit = True
            print("Correct:", correct_text, "Predict:", predict_text, "hit:", hit, "Accuracy:", hits / total)
