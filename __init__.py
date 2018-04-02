import logging
import os

from train import predict_captcha

__all__ = ["predict_captcha", "captcha_logger"]


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
