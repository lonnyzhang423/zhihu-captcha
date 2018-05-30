### 识别知乎验证码

最近在爬知乎的数据，遇到了验证码，就使用TensorFlow的CNN训练了一个能自动识别验证码的模型，最后识别线上验证码的准确率在95%左右。 


### 依赖库
1. Python3
2. PIL & numpy & requests
3. TensorFlow
4. CNN(卷积神经网络)


### 分析
爬虫请求知乎太频繁时，你会遇到两种验证码：细字体和粗字体。  
![细字体](screenshots/normal_captcha.gif)  
  
  
![粗字体](screenshots/bold_captcha.gif)  

~~如果把两种验证码混合放到同一个神经网络中训练的话，收敛会比较慢，需要的样本量就比较大，然而我并没有很多样本，只能人工打码或者买打码服务去获取样本。所以我们可以先训练一个分类器，将两种验证码区分开，再分别去训练识别，这样需要的样本量就会少很多了！**机智如我！**~~  

**UPDATE**  
收集的样本足够多啦，就不用上面那种先分类再识别的方法啦，直接把样本丢进CNN里去训练就好！


### 如何运行
1. 把你的验证码样本放到`samples`目录下的`train_mixed_captcha_base64.txt`文件中。
3. 训练模型：运行`train`目录下的`model.py`文件，直到达到你满意的准确率为止。这时训练好的网络结构和权重值等会保存在`checkpoints`目录下。
4. 在`train`目录下的`__init__.py`中恢复训练好的模型，导出`predict_captcha`预测函数。
5. 执行`predict_captcha`函数，传入base64编码的图片字符串，执行得到预测结果。
6. :boom: Boom！想干嘛你就可以干嘛了！:smirk:


### CNN
代码中的模型使用TensorFlow构建了一个简单的卷积神经网络。  
包含：一个**输入层**、三个**卷积层+池化层**和最后一个**全连接层**。  
至于网络模型为什么是这种结构，因为：**前人根据经验总结出来这种结构训练的效果会比较好**。  
当然你也可以尝试其他的结构，只要效果好就行了。我们都是**黑盒调参**的。  
  
使用**TensorBoard**可以可视化训练的相关情况。  
  
给大家看一下我训练的**网络结构**：  
  
![CNN网络结构](screenshots/graph.png)  
  
**准确率走势图**：  
  
![准确率](screenshots/accuracy.png)  
  
**loss曲线**：  
  
![loss](screenshots/loss.png)  

  
**最后**，使用训练好的模型去 **Cover** 知乎线上的验证码平均有 **90%** 的准确率，也就是两次至少命中一次的概率为：
> 1 - 0.05 * 0.05 = 0.9975  

够我用啦，然后。。。你懂得！  

**源码地址**：[https://github.com/lonnyzhang423/zhihu-captcha](https://github.com/lonnyzhang423/zhihu-captcha)  

**UPDATE**  
知乎最近升级了验证码服务，换成腾讯的滑块验证码  
之前的字符验证码样本就再次分享出来给大家学习使用吧！这下应该没人举报了吧  
关注公众号：**Python实验课** ， 后台回复：**知乎验证码** 获取百度云链接。  

**UPDATE**  
写了一个api给大家测试预测的效果  
**地址**：http://47.96.150.119/api/toolkit/captcha  
**方法**：POST  
**参数**：img_base64
