def check_prepare_args(who: type, *args):
    assert len(args) == 2, (f"准备元组参数长度需要为2，分别为类型字符串和构造关键字参数。"
                            f"{who.__name__}收到的参数数量为{len(args)}！")
    assert isinstance(args[0], str) and isinstance(args[1], dict), \
        (f"准备元组参数分别为类型字符串和构造关键字参数，"
         f"{who.__name__}收到的参数类型为{type(args[0])}和{type(args[1])}！")
    return args


net_idle_state = "空闲"
net_train_state = "训练"
net_predict_state = "预测"
net_finetune_state = "微调"
net_states = [net_idle_state, net_train_state, net_finetune_state, net_predict_state]

from .basic_nn import BasicNN


# from .trainer import Trainer
from .trainer import New2Trainer
from .net_builder import NetBuilder
from .nets import *


