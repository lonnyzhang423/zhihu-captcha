import os

from classify import classify_captcha
from predict import predict_captcha
from utils import samples_dir

__all__ = ["predict", "classify_captcha", "predict_captcha"]


def predict(image):
    clazz = classify_captcha(image)
    if clazz == 0:
        # normal captcha
        return predict_captcha(image, clazz=0)
    elif clazz == 1:
        # bold captcha
        return predict_captcha(image, clazz=1)
    else:
        # invalid captcha
        return "####"


def _eval_accuracy():
    dir = samples_dir()
    bold_file = os.path.join(dir, "bold_captcha_base64.txt")
    with open(bold_file, "r") as f:
        for line in f:
            code, image = line.split(":")
            predict_code = predict(image)
            print(predict_code)


if __name__ == '__main__':
    _eval_accuracy()
