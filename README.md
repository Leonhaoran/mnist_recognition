本仓库为作者学习神经网络而设置，学习参考B站视频https://www.bilibili.com/video/BV134421U77t

按照如下顺序运行文件即可，环境可自行配置

1.download_data.py：下载数据集

2.model.py：定义模型

3.train.py：训练模型，默认在CPU上训练

3.train_on_gpu.py：在GPU上训练模型

4.test.py：测试模型

训练模型后会将模型权重文件保存在目录下，在测试模型时会读取目录下的权重文件
batch_size=256，在GPU 3050上训练，测试准确率可以达到98.01%