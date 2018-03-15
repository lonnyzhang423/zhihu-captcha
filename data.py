import string
import io
import os
import base64
import numpy as np
from PIL import Image

__all__ = ["next_batch", "next_text_and_image", "vector2text", "img2array",
           "IMG_WIDTH", "IMG_HEIGHT",
           "CAPTCHA_FILE_NAME", "CAPTCHA_LEN", "CHAR_SET_LEN",
           "NotEnoughCaptchaException"]

numbers = string.digits
alphabets = string.ascii_lowercase
ALPHABETS = string.ascii_uppercase

CAPTCHA_LEN = 4
CHAR_SET = numbers + alphabets + ALPHABETS
CHAR_SET_LEN = len(CHAR_SET)
CAPTCHA_FILE_NAME = os.path.join(os.path.dirname(__file__), "captcha_base64.txt")

IMG_WIDTH = 150
IMG_HEIGHT = 60


class NotEnoughCaptchaException(StopIteration):
    pass


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


def img2array(image):
    if isinstance(image, str):
        img_bytes = base64.decodebytes(image.encode("utf8"))
        image = Image.open(io.BytesIO(img_bytes)).convert("L")
        image = np.array(image).flatten() / 255
    return image


captcha_file = open(CAPTCHA_FILE_NAME, "r")


def next_text_and_image():
    try:
        line = next(captcha_file)
        text, img_base64 = line.split(":")
        image = img2array(img_base64)
        return text, image
    except StopIteration:
        raise NotEnoughCaptchaException()


def next_batch(batch_size=64):
    xs = np.zeros([batch_size, IMG_WIDTH * IMG_HEIGHT])
    ys = np.zeros([batch_size, CAPTCHA_LEN * CHAR_SET_LEN])

    for i in range(batch_size):
        text, image = next_text_and_image()
        xs[i, :] = image
        ys[i, :] = text2vector(text)
    return xs, ys


if __name__ == '__main__':
    next_batch()
