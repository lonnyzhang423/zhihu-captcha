import io
import os
import base64
import logging
import numpy as np
from PIL import Image

from config import *

__all__ = ["text2vector", "vector2text", "img2array",
           "next_train_batch", "next_train_text_and_image", "next_test_text_and_image",
           "samples_dir", "checkpoints_dir", "InvalidCaptchaError"]

# 训练验证码
TRAIN_CAPTCHA = os.path.join(os.path.dirname(__file__), "samples", "train_mixed_captcha_base64.txt")
# 测试验证码
TEST_CAPTCHA = os.path.join(os.path.dirname(__file__), "samples", "test_mixed_captcha_base64.txt")


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


_train_captcha = open(TRAIN_CAPTCHA, "r")
_test_captcha = open(TEST_CAPTCHA, "r")


def next_train_text_and_image():
    global _train_captcha
    try:
        line = next(_train_captcha)
        text, img_base64 = line.split(":")[-2:]
        image = img2array(img_base64)
        return text, image
    except StopIteration:
        _train_captcha.close()
        _train_captcha = open(TRAIN_CAPTCHA, "r")
        logging.warning("Not enough captcha! Loop reading lines from same file!")
        return next_train_text_and_image()
    except InvalidCaptchaError:
        logging.warning("Invalid captcha error! Next train text and image!", exc_info=True)
        return next_train_text_and_image()


def next_test_text_and_image():
    global _test_captcha
    try:
        line = next(_test_captcha)
        clazz, text, img_base64 = line.split(":")
        image = img2array(img_base64)
        return text, image
    except StopIteration:
        _test_captcha.close()
        _test_captcha = open(TEST_CAPTCHA, "r")
        logging.warning("Not enough test captcha! Loop reading lines from same file!")
        return next_test_text_and_image()
    except InvalidCaptchaError:
        logging.warning("Invalid captcha error! Next test text and image!", exc_info=True)
        return next_test_text_and_image()


def next_train_batch(batch_size=64):
    """
    produce batch sample for prediction
    :param batch_size: default 64
    """
    xs = np.zeros([batch_size, IMG_WIDTH * IMG_HEIGHT])
    ys = np.zeros([batch_size, CAPTCHA_LEN * CHAR_SET_LEN])

    for i in range(batch_size):
        text, image = next_train_text_and_image()
        xs[i, :] = image
        ys[i, :] = text2vector(text)

    return xs, ys


def checkpoints_dir():
    return os.path.join(os.path.dirname(__file__), "train", "checkpoints")


def samples_dir():
    return os.path.join(os.path.dirname(__file__), "samples")
