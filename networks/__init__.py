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

def get_net_state(value):
    # 对用途进行状态转换
    if value == "predict" or value == net_predict_state:
        return net_predict_state
    elif value == "train" or value == net_train_state:
        return net_train_state
    elif value == "finetune" or value == net_finetune_state:
        return net_finetune_state
    elif value == "idle" or value == net_idle_state:
        return net_idle_state
    else:
        raise ValueError(f"无法识别的网络状态指示字符串，不支持将用途设置为{value}！支持的状态包括：{net_states}")


from .basic_nn import BasicNN


# from .trainer import Trainer
from .trainer import NetTrainer
from .net_builder import NetBuilder
from .nets import *


