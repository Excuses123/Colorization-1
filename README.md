# Colorization
- 简介
    - 本项目使用Keras复现论文Colorful Image Colorization内容。
    - 在Github上面找到的有质量的复现代码均为TensorFlow和PyTorch，这对很多神经网络的新人是不友好的，且前面两者的代码可读性没有Keras强，所以使用Keras复现。
- 配置
    - 本项目需要安装Keras(`pip install keras`)，默认安装tensorflow为后端，如不是，需要安装tensorflow(`pip install tensorflow`)。
- 网络搭建
    - 使用Keras Function API搭建（使用[Netron](https://lutzroeder.github.io/netron/)可视化）