import string

__all__ = ["CAPTCHA_LEN", "CHAR_SET", "CHAR_SET_LEN",
           "IMG_HEIGHT", "IMG_WIDTH",
           "NUM_PREDICT_CLASSES", "INVALID_CAPTCHA"]

numbers = string.digits
alphabets = string.ascii_lowercase
ALPHABETS = string.ascii_uppercase
INVALID_CAPTCHA = "####"

CAPTCHA_LEN = 4
CHAR_SET = numbers + alphabets + ALPHABETS
CHAR_SET_LEN = len(CHAR_SET)

IMG_WIDTH = 150
IMG_HEIGHT = 60

NUM_PREDICT_CLASSES = CAPTCHA_LEN * CHAR_SET_LEN
