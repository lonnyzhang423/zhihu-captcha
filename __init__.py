import os
import logging

from classify import classify_captcha
from predict import predict_captcha
from utils import samples_dir

__all__ = ["predict", "classify_captcha", "predict_captcha", "captcha_logger"]


def _init_captcha_logger():
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    if len(logger.handlers) > 0:
        logger.handlers.clear()

    logfile = os.path.join(os.path.dirname(__file__), "samples", "hits_captcha.log")
    fh = logging.FileHandler(logfile, encoding="utf8")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


captcha_logger = _init_captcha_logger()


def predict(image):
    clazz = classify_captcha(image)
    # clazz is 0: normal captcha
    # clazz is 1: bold captcha
    # clazz is -1: invalid captcha
    return predict_captcha(image, clazz=clazz)


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
