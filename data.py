import os
import string
import random
import numpy as np
from io import BytesIO

from PIL import Image, ImageFont, ImageOps
from PIL.ImageDraw import Draw

__all__ = ["CAPTCHA_LEN", "CHAR_SET_LEN", "IMG_WIDTH", "IMG_HEIGHT", "next_batch"]

numbers = string.digits
alphabets = string.ascii_lowercase
ALPHABETS = string.ascii_uppercase

CAPTCHA_LEN = 4
CHAR_SET = numbers + alphabets + ALPHABETS
CHAR_SET_LEN = len(CHAR_SET)

IMG_WIDTH = 150
IMG_HEIGHT = 60


class _Captcha:
    def generate(self, chars, fmt="gif"):
        im = self.generate_image(chars)
        out = BytesIO()
        im.save(out, format=fmt)
        out.seek(0)
        return out

    def write(self, chars, output, fmt="gif"):
        im = self.generate_image(chars)
        return im.save(output, format=fmt)

    def generate_image(self, chars):
        pass


class ZCaptcha(_Captcha):
    DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'fonts')
    DEFAULT_FONTS = [os.path.join(DATA_DIR, 'OpenSans-Regular.ttf')]

    def __init__(self, xy=(150, 60), fonts=None, font_sizes=None):
        self._width, self._height = xy
        self._fonts = fonts or ZCaptcha.DEFAULT_FONTS
        self._font_sizes = font_sizes or (46, 48)
        self._true_fonts = self._gen_true_fonts()

    def _gen_true_fonts(self):
        fonts = list()
        for f in self._fonts:
            if "Shadow" in f:
                # 阴影用小字体
                fonts.append(ImageFont.truetype(f, 40))
                continue
            for s in self._font_sizes:
                fonts.append(ImageFont.truetype(f, s))
        return fonts

    @property
    def truefont(self):
        return random.choice(self._true_fonts)

    def generate_image(self, chars):
        im = Image.new("L", (self._width, self._height), "white")

        draw = Draw(im)
        font = self.truefont
        shadow = (font.size == 40)

        init_offset = int(self._width / 5)
        if shadow:
            init_offset = int(self._width / 6)

        x = random.randint(0, init_offset)

        for c in chars:
            w, h = draw.textsize(c, font)
            if shadow:
                y_offset = 0
            else:
                y_offset = random.randint(10, 16)
            y = int((self._height - h) / 2 - y_offset)

            mask = Image.new("L", (w + 5, h + 5))
            Draw(mask).text((0, 0), c, fill=255, font=font)
            mask = mask.rotate(random.uniform(-30, 30), resample=Image.BILINEAR, expand=True)
            mask_colorized = ImageOps.colorize(mask, (0, 0, 0), (0, 0, 0))
            im.paste(mask_colorized, (x, y), mask)

            gap = 0
            if not shadow:
                gap = random.randint(0, 6)
            x += w - gap

        return im


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


def _random_captcha_text(char_set=CHAR_SET, captcha_len=4):
    captcha_text = []
    for _ in range(captcha_len):
        captcha_text.append(random.choice(char_set))
    return "".join(captcha_text)


captcha = ZCaptcha()


def random_captcha_text_and_image():
    text = _random_captcha_text()
    captcha_img = captcha.generate(text)
    img = Image.open(captcha_img)
    img_array = np.array(img).flatten() / 255
    return text, img_array


def next_batch(batch_size=64):
    xs = np.zeros([batch_size, IMG_WIDTH * IMG_HEIGHT])
    ys = np.zeros([batch_size, CAPTCHA_LEN * CHAR_SET_LEN])

    for i in range(batch_size):
        text, image = random_captcha_text_and_image()
        xs[i, :] = image
        ys[i, :] = text2vector(text)
    return xs, ys


if __name__ == '__main__':
    xs, ys = next_batch(1)
