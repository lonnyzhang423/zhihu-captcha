import requests
import base64
import io
import os
from PIL import Image

url = "https://www.zhihu.com/captcha.gif"
headers = {
    "User-Agent": "Chrome"
}


def collect():
    for i in range(100):
        resp = requests.get(url, headers=headers)
        bs = resp.content
        Image.open(io.BytesIO(bs)).show()
        code = input("code:")
        if code == "c":
            continue

        img_base64 = base64.encodebytes(bs)
        img_base64 = img_base64.replace(b"\n", b"")
        code = (code + ":").encode("utf8")
        line = code + img_base64 + os.linesep.encode("utf8")
        with open("bold.txt", "wb") as f:
            f.write(line)


if __name__ == '__main__':
    collect()
