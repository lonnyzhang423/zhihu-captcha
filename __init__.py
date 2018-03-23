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
    bold_file = os.path.join(dir, "test_normal_captcha_base64.txt")
    with open(bold_file, "r") as f:
        total = hits = 0
        for line in f:
            total += 1
            correct_code, image = line.split(":")
            predict_code = predict(image)
            if correct_code == predict_code:
                hits += 1
            print("Correct:", correct_code, "Predict:", predict_code)
        print("Accuracy:", hits / total)


if __name__ == '__main__':
    _eval_accuracy()
