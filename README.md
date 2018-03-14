# Zhihu-captcha
Crack zhihu captcha with TensorFlow

# Prerequisites
1. Python3
2. PIL
3. TensorFlow
4. CNN(Convolutional Neural Network)

# Steps
1. We use Python3 and TensorFlow to build a CNN model
2. And then construct [zhihu-like](https://www.zhihu.com/captcha.gif) captcha image with PIL
3. Next feed random self-generated [zhihu-like](https://www.zhihu.com/captcha.gif) captcha image into the prebuilt network
4. Train the network until you are satisfied with the predict accuracy,and save the network
5. Restore the network,feed the real zhihu captcha image and get predict captcha
6. Finished!
