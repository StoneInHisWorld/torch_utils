# torch_utils
根据pytorch编写的网络框架以及运行支持工具包，使用本框架可以大大减少代码编写量。

## ControlPanel
`from utils.hypa_control import ControlPanel`  
控制台类。用于读取运行参数设置，设定训练超参数以及自动编写日志文件等一系列与网络构建无关的操作。
1. 使用`cp = ControlPanel()`初始化
   1. 使用`for`得到每次训练使用的训练器`Trainer`，单个`Trainer`表示使用一组待训练超参数
   2. 对训练器使用`with`语句即可得到本次训练使用的超参数。此时，训练开始计时。退出`with`语句块后，计时结束，并编写日志。  
   示例：
       ```
        for trainer in cp:
            with trainer as hps:
                epochs, batch_size, ls_fn, lr = hps
       ```
2. 得到每组训练超参数对应的`Trainer`前，会读取一次运行配置参数。
   1. 该运行配置参数可以动态更改，每次更改将在下一组超参数训练中生效。 
   2. 实验编号`exp_no`属于运行配置参数，将随着`ControlPanel`对象的迭代而递增，即在读取运行配置参数时递增。
3. 运行配置参数可以通过字典形式访问
   1. 示例：`device = cp['device']`
   2. 运行设备`device`属于运行配置参数，可以使用属性访问符直接进行访问。e.g. `device = cp.device`
   3. 实验编号`exp_no`属于运行配置参数，可以使用属性访问符直接进行访问。e.g. `exp_no = cp.exp_no`
4. 内置绘制历史函数`plot_history()`，会对训练过程中历史损失值、准确率进行趋势图绘制
   1. 需要主动调用才会绘制图像，将在未来的版本中自动进行调用
5. 每次训练器训练完毕后会自动编写日志，或处理和记录出错信息。
   1. 使用`add_logMsg()`函数增加单条日志的信息项。
   2. 每条日志都有独立的编号，从小到大排序。
### TODO
1. 未来将加入GUI用于调整超参数。
2. 未来将编写编程示例。
## networks包
### networks.layers
神经网络通用层，可用于构造自定义结构的神经网络模型。
1. `MultiOutputLayer`：多通道输出层。将单通道输入扩展为多通道输出。
2. `Reshape`：重塑层，可以将输入的张量进行重塑为所需形状。
3. `SSIMLoss`：SSIM损失层。计算每对y_hat与y的图片结构相似度，并求其平均逆作为损失值。计算公式为：$$ loss = \frac{\sum_{i=1}^n 1 - ssim}{n} $$ 其中，$ssim$为单个特征-标签对求出的结构相似度，$n$为样本数。
4. `Val2Fig`: 数值-图片转化层。根据指定模式，对数值进行归一化后反归一化为图片模式像素取值范围，从而转化为可视图片。
5. 持续添加中……
### networks.nets
包含各种神经网络模型，均继承自`BasicNN`
1. `AlexNet`: 经典AlexNet模型。  
`from networks.nets.alexnet import AlexNet as NET`
2. `GoogLeNet`: 经典GoogLeNet模型。  
`from networks.nets.googlenet import GoogLeNet as NET`
3. `LeNet`: 经典LeNet模型。  
`from networks.nets.lenet import LeNet as NET`
4. `MLP`: 经典多层感知机。  
`from networks.nets.mlp import MLP as NET`
5. `Pix2Pix`: 适用于图片翻译、转换任务的学习模型。  
参考论文：  
[1] 王志远. 基于深度学习的散斑光场信息恢复[D]. 厦门：华侨大学，2023  
[2] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou and Alexei A. Efros.
   Image-to-Image Translation with Conditional Adversarial Networks[J].
   CVF, 2017. 1125, 1134  
`from networks.nets.pix2pix import Pix2Pix as NET`
6. `SLP`: 经典单层感知机。  
`from networks.nets.slp import SLP as NET`
7. `VGG`: 经典VGG网络模型，可通过指定conv_arch构造指定版本的VGG网络。  
`from networks.nets.vgg import VGG as NET`
8. `WZYNetEssay`: 通过不断卷积，下采样，提取图片信息的网络。  
参考：  
[1] 王志远. 基于深度学习的散斑光场信息恢复[D]. 厦门：华侨大学，2023  
`from networks.nets.wzynet_essay import WZYNetEssay as NET`
9. 持续添加中……
### networks.basic_nn
内置基础网络模型`BasicNN`，继承自`torch.nn.Sequential`，提供神经网络的基本功能，包括权重初始化，训练以及测试。
### example.py
尚未完成编辑，请不要使用！
