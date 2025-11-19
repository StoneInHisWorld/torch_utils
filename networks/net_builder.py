import inspect
import warnings

from torchsummary import summary

from networks import BasicNN, configure_network


def get_signature(module):
    module_sig = inspect.signature(module)
    # 分类参数
    positional_only = []  # 仅位置参数
    positional_or_keyword = []  # 位置或关键字参数
    keyword_only = []  # 仅关键字参数
    needed = []

    for param_name, param in module_sig.parameters.items():
        kind, default = param.kind, param.default
        if kind == inspect.Parameter.VAR_KEYWORD:
            continue
        elif kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        elif kind == inspect.Parameter.POSITIONAL_ONLY:
            # positional_only.append((param_name, default))
            positional_only.append(param_name)
        elif kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            # positional_or_keyword.append((param_name, default))
            positional_or_keyword.append(param_name)
        elif kind == inspect.Parameter.KEYWORD_ONLY:
            # keyword_only.append((param_name, default))
            keyword_only.append(param_name)
        if default == inspect.Parameter.empty:
            needed.append(param_name)
    return positional_only, positional_or_keyword, keyword_only, needed

class NetBuilder:
    """
    NetBuilder负责保存BasicNN的位置参数和关键字参数，以及准备参数

    Args:


    Usage:
        nb = NetBuilder(*base_args, **base_kwargs)
        nb.set_optimizer_args('sub1', lr=1e-3, weight_decay=1e-5)
        nb.set_lr_scheduler_args('sub1', step_size=10, gamma=0.1)
        args, kwargs = nb.prepare_args('sub1')
        model = nb.build(BasicNNClass, 'sub1') # equivalent to BasicNNClass(*args, **kwargs)
    """

    def __init__(self, module, config, *init_args, **init_kwargs):
        # Validate that init_args or init_kwargs are provided
        self.module = module
        self.config = config
        m_po, m_pok, m_ko, m_needed = get_signature(module)
        _, bn_pok, bn_ko, _ = get_signature(BasicNN)
        n_valid_argments = len(init_args) + len(init_kwargs.keys() - [b[0] for b in bn_ko])
        if n_valid_argments < len(m_needed):
            msg = (f"输入的参数不完整！{module.__name__}的构造参数包括位置参数{m_po}, 位置/关键字参数{m_pok}, "
                   f"关键字参数{m_ko}。\n{BasicNN.__name__}可以输入的参数包括{bn_ko}。\n"
                   f"创建{module.__name__}必需的参数为{m_needed}。")
            raise ValueError(msg)

        assert issubclass(module, BasicNN), f'{self.__class__.__name__}只服务BasicNN的子类！'
        self.module = module
        self.init_args = init_args
        init_kwargs['with_checkpoint'] = self.config['with_checkpoint']
        self.init_kwargs = init_kwargs

        # # per-submodule argument storages
        # # each maps sub_key -> (positional_args_tuple, keyword_args_dict)
        # self._optimizer_args = {}
        # self._lr_scheduler_args = {}
        # self._training_ls_fn_args = {}
        # self._testing_ls_fn_args = {}

    # Setter methods for per-submodule argument groups
    def set_optimizer_args(self, *args):
        """保存模型的优化器参数"""
        self._optimizer_args = args

    def set_lr_scheduler_args(self, *args):
        """保存模型的学习率规划期参数"""
        self._lr_scheduler_args = args

    def set_training_ls_fn_args(self, *args):
        """保存模型的训练损失函数参数"""
        self._training_ls_fn_args = args

    def set_testing_ls_fn_args(self, *args):
        """保存模型的测试损失函数参数"""
        self._testing_ls_fn_args = args

    def build(self, is_train: bool, mute: bool = False):
        """根据参数构造一个神经网络
        :param net_class: 构造的神经网络类型，作为类构造器传入
        :param n_init_args: 类构造器所用位置参数
        :param n_init_kwargs: 类构造器所用关键字参数
        :return: 构造完成的神经网络
        """
        if not mute:
            print(f'\r正在构造{self.module.__name__}', end='', flush=True)
        net = self.module(*self.init_args, **self.init_kwargs)
        if not mute:
            print(f'\r构造{self.module.__name__}完成')
            self.__list_net(net)
        assert hasattr(self, '_optimizer_args'), (f"没有为网络配置优化器参数，请通过调用{self.__class__.__name__}()."
                                                  f"{self.set_optimizer_args.__name__}()方法进行配置！")
        assert hasattr(self, '_lr_scheduler_args'), (f"没有为网络配置学习率规划器参数，请通过调用{self.__class__.__name__}()."
                                                     f"{self.set_lr_scheduler_args.__name__}()方法进行配置！")
        assert hasattr(self, '_training_ls_fn_args'), (f"没有为网络配置训练损失函数参数，请通过调用{self.__class__.__name__}()."
                                                       f"{self.set_training_ls_fn_args.__name__}()方法进行配置！")
        assert hasattr(self, '_testing_ls_fn_args'), (f"没有为网络配置测试损失函数参数，请通过调用{self.__class__.__name__}()."
                                                      f"{self.set_testing_ls_fn_args.__name__}()方法进行配置！")
        self.activate_model(net, is_train, mute)
        return net

    def activate_model(self, net, is_train: bool = True, mute: bool = False):
        """<UNK>
        :param net: <UNK>
        """
        configure_network(
            net, is_train, mute, self._optimizer_args, self._lr_scheduler_args,
            self._training_ls_fn_args, self._testing_ls_fn_args
        )

    def __list_net(self, net) -> None:
        """打印网络信息。
        :param net: 待打印的网络
        :return: None
        """
        if self.config['print_net']:
            input_size = net.input_size
            if input_size:
                try:
                    summary(net, input_size=(self.config['batch_size'], *input_size), device=net.device)
                    return
                except Exception as e:
                    warnings.warn(f"打印网络时遇到错误：{e}，只显示网络结构！")
            else:
                warnings.warn(f"输入形状{input_size}无法解析，只显示网络结构！")
            print(net)