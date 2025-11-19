from tqdm import tqdm

from networks.decorators import prepare
from data_related import SelfDefinedDataSet
from utils.accumulator import Accumulator
from utils.func import pytools as ptools
from utils.history import History

from .__log_impl import log_impl
from .__train_impl import train_impl, tv_impl, train_with_k_fold
from .__train_impl import tv_multiprocessing_impl as tv_multiprocessing


def is_multiprocessing(n_workers):
    return n_workers >= 3


def prepare_test(fn):
    """
    Decorator for training preparation and cleanup.
    Allows user-defined operations before and after training.
    Usage:
        @prepare_train
        def train_fn(...): ...
    """

    def wrapper(trainer, test_iter):
        # 创建网络
        assert hasattr(trainer, 'module'), "训练器中不含模型对象，是否是尚未训练模型？"
        trainer.net_builder.activate_model(trainer.module, False)
        # 设置进度条
        setattr(trainer, "pbar", tqdm(
            total=len(test_iter), unit='批', desc=f'\r正在进行测试准备……', ncols=100,
            bar_format='{desc}{n}/{total} | {elapsed}/{remaining} | {rate_fmt}{postfix}'
        ))
        # test_iter = tqdm(test_iter, unit='批', position=0, desc=f'\r测试中……', mininterval=1, ncols=100)
        result = fn(trainer, test_iter)
        trainer.module.deactivate()
        trainer.pbar.set_description("\r测试完毕")
        trainer.pbar.close()
        del trainer.pbar
        return result

    return wrapper


class Trainer:
    """神经网络训练器对象，提供所有针对神经网络的操作，包括训练、验证、测试、预测"""

    def __init__(
        self, net_builder, criterion_a,
        runtime_cfg, hps=None  # 训练、验证、测试依赖参数
    ):
        """神经网络训练器对象，提供所有针对神经网络的操作，包括训练、验证、测试、预测。
        Trainer类负责进行网络构建以及网络训练方法实现，可以通过Trainer.module获取训练完成或正在训练的网络对象。

        :param module_class: 模型类，用于调用构造函数实现网络构造
        :param m_init_args: 网络构造位置参数
        :param m_init_kwargs: 网络构造关键字参数
        :param prepare_args: 网络训练准备参数，用于BasicNN中的prepare_training()函数
        :param criterion_a: 评价指标计算方法，如需指定多个请传入`list`对象
        :param hps: 超参数组合，用于获取其中的超参数进行训练
        :param runtime_cfg: 动态运行参数，用于获取其中的参数指导训练
        """
        if hps is None:
            hps = {}
        # 设置训练、验证、测试依赖参数
        self.hps = hps
        self.runtime_cfg = runtime_cfg
        self.criterion_a = criterion_a if isinstance(criterion_a, list) else [criterion_a]
        if len(self.criterion_a) == 0:
            raise ValueError(f"训练器没有拿到训练指标方法!"
                             f"请检查自定义数据集类{SelfDefinedDataSet.__name__}中对于"
                             f"{net_builder.module.__name__}的评价指标赋值。")
        self.pbar = None
        self.net_builder = net_builder

    def train(self, data_iter) -> History:
        """训练公共接口。
        拆解数据迭代器，并根据训练器超参数以及动态运行参数判断进行的训练类型，调用相应训练函数。
        训练进度条在此创建。

        :param data_iter: 训练所用数据迭代器
        :return: 训练数据记录对象
        """
        # 提取所需超参数以及动态运行参数
        self.k = self.hps['k']

        # 判断是否是k折训练
        if self.k > 1:
            train_fn, train_args = train_with_k_fold, (self, data_iter)
        else:
            # 提取训练迭代器和验证迭代器
            data_iters = [it[0] for it in data_iter]
            # if len(data_iter) == 2:
            #     train_iter, valid_iter = [it[0] for it in data_iter]
            #     pbar_len = (len(train_iter) + len(valid_iter)) * n_epochs
            # elif len(data_iter) == 1:
            #     train_iter, valid_iter = data_iter[0][0], None
            #     pbar_len = len(train_iter) * n_epochs
            # else:
            #     raise ValueError(f"无法识别的数据迭代器，其提供的长度为{len(data_iter)}")
            # 判断是否要进行多线程训练
            if not is_multiprocessing(self.runtime_cfg['n_workers']):
                # 不启用多线程训练
                if len(data_iters) == 2:
                    # 进行训练和验证
                    train_fn = tv_impl
                elif len(data_iters) == 1:
                    # 进行训练
                    train_fn = train_impl
                else:
                    raise ValueError(f"无法识别的数据迭代器，其提供的长度为{len(data_iters)}")
                train_args = (self, *data_iters)
            else:
                # 启用多进程训练
                train_fn, train_args = tv_multiprocessing, (self, *data_iters)
        history = train_fn(*train_args)
        return history

    @prepare('predict')
    def predict(self, ret_ls_metric=True, ret_ds=True):
        """根据参数创建网络并进行预测
        对于数据迭代器中的每一批次数据，根据需要保存输入数据、预测数据、标签集、评价指标、损失值数据，并打包返回。

        Args:
            ret_ls_metric: 是否返回损失值数据以及评价指标数据
            ret_ds: 是否返回原数据集

        Returns:
            [预测数据]
             + [特征数据集，标签数据集] if ret_ls_metric == True
             + [评价指标数据，损失值数据，评价指标函数名称，
                损失值函数名称] if ret_ds == True
        """
        # 提取训练器参数
        net = self.module
        criterion_a = self.criterion_a
        pbar = self.pbar
        # 本次预测所产生的全部数据池
        inputs, predictions, labels, metrics, loss_pool = [], [], [], [], []
        # 对每个批次进行预测，并进行评价指标和损失值的计算
        for fe_batch, lb_batch in pbar:
            result = net.forward_backward(fe_batch, lb_batch, False)
            pre_batch, ls_es = result
            predictions.append(pre_batch)
            if ret_ds:
                inputs.append(fe_batch)
                labels.append(lb_batch)
            if ret_ls_metric:
                metrics.append([
                    criterion(pre_batch, lb_batch, size_averaged=False)
                    for criterion in criterion_a
                ])
                loss_pool.append(ls_es)
        pbar.set_description('结果计算完成')
        # # 将所有批次的数据堆叠在一起
        # ret = [torch.cat(predictions, dim=0)]
        # if ret_ds:
        #     ret.append(torch.cat(inputs, dim=0))
        #     ret.append(torch.cat(labels, dim=0))
        # if ret_ls_metric:
        #     ret.append(torch.cat(metrics, dim=0))
        #     ret.append(torch.cat(loss_pool, dim=0))
        #     ret.append([
        #         ptools.get_computer_name(criterion) for criterion in criterion_a
        #     ])
        #     ret.append(net.test_ls_names)
        # return ret
        ret = [predictions]
        if ret_ds:
            ret += [inputs, labels]
        if ret_ls_metric:
            ret += [metrics, loss_pool, [
                ptools.get_computer_name(criterion) for criterion in criterion_a
            ], net.test_ls_names]
        return ret

    @prepare_test
    def test(self, test_iter) -> dict:
        """测试实现。
        每次取出测试数据供给器中的下一批次数据进行前向传播，计算评价指标和损失，记录到日志中。

        :param test_iter: 测试数据迭代器
        :return: 测试记录
        """
        # 提取训练器参数
        net = self.module
        criterion_a = self.criterion_a
        # 要统计的数据种类数目
        l_names = [f'test_{item}' for item in net.test_ls_names]
        metric_acc = Accumulator(len(criterion_a) + len(l_names) + 1)
        c_names = [f'test_{ptools.get_computer_name(criterion)}' for criterion in criterion_a]
        # 计算准确率和损失值
        for features, labels in test_iter:
            preds, ls_es = net.forward_backward(features, labels, False)
            # num_samples = len(preds)
            # metric_acc.add(
            #     *[criterion(preds, labels) for criterion in criterion_a],
            #     *[ls * num_samples for ls in ls_es], len(preds)
            # )
            log_impl(
                self.pbar, preds, labels, ls_es, l_names,
                criterion_a, c_names, metric_acc
            )
        # 生成测试日志
        log = {}
        i = 0
        for i, (computer, name) in enumerate(zip(criterion_a, c_names)):
            log[name] = metric_acc[i] / metric_acc[-1]
        i += 1
        for j, ln in enumerate(l_names):
            log[ln] = metric_acc[i + j] / metric_acc[-1]
        return log
