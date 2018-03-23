import io
import os
import base64
import logging
import numpy as np
from PIL import Image

from config import *

__all__ = ["text2vector", "vector2text", "img2array",
           "next_normal_text_and_image", "next_predict_batch", "next_classify_batch",
           "classify_checkpoints_dir", "samples_dir",
           "predict_normal_checkpoints_dir", "predict_bold_checkpoints_dir",
           "InvalidCaptchaError"]

# 细体验证码
os.path.dirname(__file__)
NORMAL_CAPTCHA = os.path.join(os.path.dirname(__file__), "samples/normal_captcha_base64.txt")
# 粗体验证码
BOLD_CAPTCHA = os.path.join(os.path.dirname(__file__), "samples/bold_captcha_base64.txt")


def text2vector(text):
    text_len = len(text)
    if text_len > CAPTCHA_LEN:
        raise ValueError("Max captcha is 4 chars!")

    vector = np.zeros(CHAR_SET_LEN * CAPTCHA_LEN)

    def char2pos(c):
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError("No map!")
        return k

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector


def vector2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


class InvalidCaptchaError(OSError):
    pass


def img2array(image):
    try:
        if isinstance(image, str):
            img_bytes = base64.decodebytes(image.encode("utf8"))
            image = Image.open(io.BytesIO(img_bytes)).convert("L")
            image = np.array(image).flatten() / 255
        return image
    except OSError:
        raise InvalidCaptchaError()


_normal_captcha_file = open(NORMAL_CAPTCHA, "r")
_bold_captcha_file = open(BOLD_CAPTCHA, "r")


def next_normal_text_and_image():
    global _normal_captcha_file
    try:
        line = next(_normal_captcha_file)
        text, img_base64 = line.split(":")
        image = img2array(img_base64)
        return text, image
    except StopIteration:
        _normal_captcha_file.close()
        _normal_captcha_file = open(NORMAL_CAPTCHA, "r")
        logging.warning("Not enough normal captcha! Loop reading lines from same file!")
        return next_normal_text_and_image()


def next_normal_image():
    _, image = next_normal_text_and_image()
    return image


def next_bold_text_and_image():
    global _bold_captcha_file
    try:
        line = next(_bold_captcha_file)
        text, img_base64 = line.split(":")
        image = img2array(img_base64)
        return text, image
    except StopIteration:
        _bold_captcha_file.close()
        _bold_captcha_file = open(NORMAL_CAPTCHA, "r")
        logging.warning("Not enough bold captcha! Loop reading lines from same file!")
        return next_normal_text_and_image()


def next_bold_image():
    _, image = next_bold_text_and_image()
    return image


def next_predict_batch(batch_size=64, clazz=0):
    """
    produce batch sample for prediction
    :param batch_size: default 64
    :param clazz:
                  0: normal captcha
                  1: bold captcha
    """
    xs = np.zeros([batch_size, IMG_WIDTH * IMG_HEIGHT])
    ys = np.zeros([batch_size, CAPTCHA_LEN * CHAR_SET_LEN])

    for i in range(batch_size):
        if clazz == 0:
            text, image = next_normal_text_and_image()
        else:
            text, image = next_bold_text_and_image()
        xs[i, :] = image
        ys[i, :] = text2vector(text)
    return xs, ys


def next_classify_batch(batch_size=64):
    """
    produce batch samples for classification
    """
    xs = np.zeros([batch_size, IMG_WIDTH * IMG_HEIGHT])
    ys = np.zeros([batch_size, NUM_CLASSIFY_CLASSES])

    for i in range(batch_size):
        if i % 2 == 0:
            image = next_normal_image()
            y = [1, 0]
        else:
            image = next_bold_image()
            y = [0, 1]
        xs[i, :] = image
        ys[i, :] = y
    return xs, ys


def predict_normal_checkpoints_dir():
    return os.path.join(os.path.dirname(__file__), "predict/normal_checkpoints")


def predict_bold_checkpoints_dir():
    return os.path.join(os.path.dirname(__file__), "predict/bold_checkpoints")


def classify_checkpoints_dir():
    return os.path.join(os.path.dirname(__file__), "classify/checkpoints")


def samples_dir():
    return os.path.join(os.path.dirname(__file__), "samples")
