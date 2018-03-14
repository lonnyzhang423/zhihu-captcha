# Zhihu-captcha
Crack zhihu captcha with TensorFlow

# Prerequisites
1. Python3
2. PIL & numpy
3. TensorFlow
4. CNN(Convolutional Neural Network)

# Steps
1. We use Python3 and TensorFlow to build a CNN model
2. For better accuracy,mark as many as possible captcha images and save to your disk
3. Then convert your marked [zhihu captcha](https://www.zhihu.com/captcha.gif) to image and label array with PIL & numpy
4. Next feed the image and label array into the prebuilt network,train the network until you are satisfied with the predict accuracy
5. Save your well trained network model
6. Restore the network model,feed the production-environment [zhihu captcha](https://www.zhihu.com/captcha.gif) and predict
7. Cracked!

# FAQ
