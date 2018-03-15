[English](README.md)
### 知乎验证码
使用TensorFlow破解知乎验证码

### 前提
1. Python3
2. PIL库、numpy库
3. TensorFlow使用
4. CNN(卷积神经网络)

### 步骤
1. 使用Python3和TensorFlow构建一个卷积神经网络
2. 为了提高识别准确率，尽可能多的标记知乎[验证码](https://www.zhihu.com/captcha.gif)，并存储到本地
3. 使用PIL和numpy库将标记好的验证码转化成图片和标签数组
4. 把图片和标签数据feed到卷积神经网络中，不断的训练，直到达到满意的准确率
5. 保存训练好的网络模型
6. 恢复网络模型，拉取知乎线上验证码转换成图片数组feed到神经网络中进行预测
7. Boom!:boom:
