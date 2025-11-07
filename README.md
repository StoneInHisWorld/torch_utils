# torch_utils v0.4
基于`pytorch`框架的训练框架工具包，包括数据处理工具、编写好的网络架构和网络层以及实用的python工具。  
详细的使用方法参见给出的编程范例以及每个类和方法的pydoc。

## 更新日志
### _v0.5_（即将推出）
1. 添加了`ResNet50`以及`VisionTransformer`
### _v0.4_
1. `settings.json`文件现在包含了训练器、自定义数据集以及数据加载器的关键字参数，`checkpoint`以及`hook`功能都将在这里进行开关。
2. 完善了多线程数据读取、多线程数据预处理、数据集单例预处理、懒加载数据集以及这些功能的组合使用。
3. 修复了`pix2pix`无法注入参数的bug。
4. 添加了`StorageDataLoader`以及`DataTransformer`来处理存储读取数据以及数据预处理。
5. 所有配置文件现在都配有默认结构和值，都能够在没有找到的情况下自动创建。
6. 编写好了适配新架构的编程实例。
### _v0.3_
1. 加入了`torch.profile.profiler`的支持，可以通过调用`Trainer().train_with_profiler()`来对网络进行性能评测，
可以在log/profiling中，根据时间找到相应的训练数据。
2. 各网络运行文件中，简化了数据迭代器的参数指定。
3. 添加了懒加载数据集、懒加载数据迭代器的支持，使用懒加载可以使得内存要求大大降低。
4. 将`data_related.criteria`模块中的评价指标算法调用换成了CUDA可加速版本，大大提高了CUDA设备上的运行速度。
### _v0.2_
1. 完整的`README.md`编写完毕。
2. 使用装饰器完成网络创建和训练准备的训练器`Trainer`。
3. 更专注于网络架构本身的`BasicNN`。
4. 编程实例编写完毕。
5. 高度抽象的DAO数据集类`SelfDefinedDataSet`。
### _v0.1_
第一个可运行版本。

### _未来更新_
1. 支持多进程训练验证。
2. 支持GUI操作。
3. 完全覆盖的中文提示。
4. 英文语言包支持。

## config
- 定义所有配置文件的初始架构和默认值
- 运行本框架依赖的库详见`./config/torch_env.yml`。

## data_related
`import data_related as dr`  
数据处理工具包，包括评价指标计算方法、数据集及其操作实现。
1. `from dr.criteria import *`  
提供评价指标计算方法。
   - 如提供的评价指标函数没办法满足需求，请在自定义数据集文件中定义您的评价指标函数。   
   - 自定义的评级指标函数需要满足接口：
```python
def PCC(Y_HAT, Y, size_averaged: bool = True):
    """
    :param Y_HAT: 预测值
    :param Y: 标签值
    :param size_averaged: 是否需要进行批平均
    :return 计算好的评价指标
    """
```
2. `from dr.data_transformer import DataTransformer`  
数据转换器，用于从存储取出数据后对数据进行预处理
- 用户需要继承此类，来定义自己的预处理程序。
3. `from dr.dataloader import *`  
数据加载器实现，从`DataSet`对象中取出数据对并经过`transit_fn`进行数据迁移后，返回给调用者。
   - 懒数据加载器则是通过索引供给器从数据集中取出数据，经过预处理和数据迁移后返回给调用者。
   - `transit_fn`需要有如下接口：
```python
def transit_fn(batch, **kwargs):
    """
    :param batch: 需要预处理的数据批
    :param kwargs: 预处理所用关键字参数
    :return: 数据批次
    """
```
3. `from dr.datasets import *`  
数据集实现，持有原始数据集同时负责进行数据预处理调用。
4. `from dr.ds_operation import *`
提供数据集级别的操作，包括k折分割、数据加载器转化、数据集切片、数据集正则化和逆正则化。
6. `from dr.SelfDefinedDataSet import SelfDefinedDataSet`  
用户自定义数据集，负责读取原始数据、数据集切片、展示图片保存、数据集转换。
- 该类为用户编辑类，需要实现相关抽象方法以完成上述功能。
7. `from dr.storage_dloader import StorageDataLoader`
处理线程设置以及进度条显示设置，实现存储数据读取
- 用户需要继承该类以实现自定义的存储数据读取

## networks
`import networks`  
网络包，包含有已实现的、可调用的神经网络层与网络模块。
1. `import networks.basic_nn`  
基本神经网络类，提供神经网络的基本功能，包括训练准备（优化器生成、学习率规划器生成、损失函数生成）、模块初始化、前反向传播实现以及展示图片输出注释的实现。
2. `import networks.trainer`  
神经网络训练器，提供所有针对神经网络的操作，包括神经网络创建、训练、验证、测试以及预测。
3. `layers`  
`from networks.layers import *`  
包含神经网络通用层及其依赖方法，可用于构造自定义结构的神经网络模型，所有网络层均继承于`torch.nn.Module`。
4. `nets`  
`from networks.nets import *`  
包含神经网络及其依赖方法，可直接用于训练任务。所有神经网络均继承于`from networks.basic_nn import BasicNN`。

## utils
`import utils`
实用工具包，提供以上所有包依赖的实用工具。基于`pytorch`、`pandas`、`numpy`、`PIL`库实现。
1. `func`  
提供实用工具函数。
   1. `import utils.func.img_tools as itools`  
   图片工具包，包括图片重塑、裁剪、读取、二值化、拼接、添加掩膜等一系列操作，依赖于`PIL`、`numpy`编写
   2. `import utils.func.log_tools as ltools`  
   日志工具包，包括日志编写、日志读取以及历史趋势图绘制功能，依赖于`matplotlib`编写。
   3. `import utils.func.pytools as ptools`  
   基于python内置库的工具包，包括路径检查、超参数列表生成、参数合法性判断以及多线程处理功能。
   4. `import utils.func.tensor_tools as tstools`  
   `pytorch`张量工具包，提供张量与PIL图片的互相转换功能。
   5. `import utils.func.torch_tools as ttools`
   `pytorch`工具包，提供设备探测、优化器获取、损失函数获取、权重偏移初始化、学习率规划器获取以及激活函数获取功能。
2. `from utils.accumulator import Accumulator`  
浮点数累加器，负责训练过程中的浮点数指标的累加。
3. `from utils.ctrl_panel import ControlPanel`  
控制台类负责读取、管理动态运行参数、超参数组合，以及实验对象的提供。
4. `import utils.decorators`  
装饰器模块，包括训练器依赖的`prepare`、`net_builder`装饰器以及其他装饰器，负责解决核心功能外的预处理和后处理。
5. `from utils.experiment import Experiment`  
实验对象，负责神经网络训练的相关周边操作，计时、显存监控、日志编写、网络持久化、历史趋势图绘制及保存。
6. `from utils.history import History`  
历史记录器，以数值列表的形式存储在对应名称属性中。
7. `from utils.history import Process`  
进程对象，继承于`torch.multiprocessing.Process`，应用于多进程处理任务如多进程训练。
8. `from utils.history import Thread`  
线程对象，继承于`threading.Thread`，应用于多线程处理任务如数据集预处理。

## examples
提供编程范例以快速应用本工具包至具体项目。`example_project_structure`为示例项目结构，其中有几个文件需要编辑：
1. `main_example.py`  
主程序，调用所有接口以进行神经网络对象的训练、验证与测试。
2. `trained_example.py`  
此文件用于使用训练完成的网络模型可视化输出结果。
3. `self-defined-ds_example.py`  
此文件用于定义存储接触数据集，读取源数据。
4. `example_hp.json`  
此文件用于为训练过程指定超参数。
5. `README.md`  
编辑项目说明。  

## settings.json
`settings.json`包括了所有目前支持的运行动态参数，以及训练器、自定义数据集和数据加载器的关键字参数。
- 可以更改其中的参数值来适配您的设备和项目结构
- 该配置文件的读取使用[jsonref](https://jsonref.readthedocs.io/en/latest/#)实现，如需了解带有引用的json参数如何编写，请参考官方文档

### _hook机制_
**hook机制**会在`BasicNN`网络中，逐层注册输入、梯度抓取函数。对于`BasicNN`网络的前后两次前向传播和反向传播，
比较两次前向传播的输入输出是否相等，前后两次反向传播的梯度输入输出梯度是否相等，并将比较结果输出到命令行中。
***使用本机制可以检查梯度消失以及求导出错问题，但会增加训练计算、内存消耗。***  
`with_hook`设置为`True`即可开启pytorch的**hook机制**，`hook_mute`参数设置为`True`，可用于简化**hook机制**输出，屏蔽不必关心的输入或梯度消息。  
该机制仅作用于`Trainer.__train()`、`Trainer.__train_and_valid()`函数。  

### _checkpoint机制_
**checkpoint机制**（**检查点机制**）作用于`BasicNN`网络的`__call__()`函数中，使用`checkpoint.checkpoint()`进行网络的前向传播。
该机制作用时，`BasicNN`网络的前向传播不会保存结果，反向传播时会再进行一次前向传播以进行梯度计算。
***使用本机制理论上可以大大降低显存要求，但会增加训练计算量。***  
`with_checkpoint`设置为`True`即可开启pytorch的**checkpoint机制**