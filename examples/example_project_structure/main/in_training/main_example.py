import sys

sys.path.append('where/you/put/torch_utils')
sys.path.append('where/you/put/project')

from networks.trainer import Trainer
from data_related import ds_operation as dso
from utils.ctrl_panel import ControlPanel
# TODO: 选择读取数据集的类
from your_DAO_module import your_DAO_class as DataSet
# TODO: 选择训练的网络架构类型
from your_net_module import your_Net as Net

"""此文件用于使用超参数组合进行神经网络对象的训练、验证与测试"""
net_name = Net.__name__.lower()
dataset_name = DataSet.__class__.__name__


def transit_fn(batch, **kwargs):
    """可自定义的数据供给传输函数
    DataLoader每次从内存取出数据后，都会调用本函数对批次进行预处理

    :param batch: 需要预处理的数据批
    :param kwargs: 预处理所用关键字参数
    :return: 数据批次
    """
    ...


if __name__ == '__main__':
    # 调参面板
    # TODO：设置控制面板路径参数
    cp = ControlPanel(
        DataSet,  # 处理的数据集类
        f'your/hyper_parameters/path.json',  # 超参数配置json文件路径
        f'your/runtime_cfg/path.json',  # 运行配置json文件路径
        f'your/log/path.json',  # 结果存储文件路径
        f'where/you/put/your/trained_net',  # 训练成果网络存储路径
        f'where/you/put/your/history_plot'  # 历史趋势图存储路径
    )

    print('正在整理数据……')
    # TODO: 设置数据集创建参数
    which_dataset = 'choose which dataset under your data root'
    data = DataSet('where/you/put/your/dataset', which_dataset, Net, ...)
    criterion_a = DataSet.get_criterion_a()

    print('数据预处理中……')
    train_ds, test_ds = data.to_dataset()
    del data

    # 多组参数训练流水线
    for experiment in cp:
        with experiment as hps:
            # TODO：在此处定义传入给transit_fn的关键字参数
            transit_kwargs = {...}
            # 使用k-fold机制
            data_iter_generator = (
                [
                    # TODO：在此处定义训练数据供给器参数
                    dso.to_loader(train_ds, transit_fn, ...)
                    for sampler in sampler_group
                ]
                # TODO: 设置数据集分割参数
                for sampler_group in dso.split_data(train_ds, ...)
            )  # 将抽取器遍历，构造加载器
            # 获取数据迭代器并注册数据预处理函数
            # TODO：在此处定义测试数据供给器参数
            test_iter = dso.to_loader(test_ds, transit_fn, ...)
            # TODO：设置网络构建参数，按照网络的初始化参数进行指定
            net_init_kwargs = {...}
            net_init_args = (...,)
            # 进行训练准备
            # TODO：在此处设置网络训练参数。训练位置参数有三个，按位序为优化器参数、学习率规划器参数、损失函数参数。
            # 优化器参数要求为二元组列表，二元组内容为（优化器类型字符串，优化器构造关键字参数）
            # 学习率规划器参数要求为二元组列表，二元组内容为（学习率规划器类型字符串，学习率规划器构造关键字参数）
            # 损失函数参数要求为二元组列表，二元组内容为（损失函数类型字符串，损失函数构造关键字参数）
            prepare_args = ([...], [...], [...])
            trainer = Trainer(
                Net, net_init_args, net_init_kwargs, prepare_args,
                DataSet, criterion_a, hps, cp.runtime_cfg
            )
            # 开始训练、验证和测试
            train_log = trainer.train(data_iter_generator)
            test_log = trainer.test(test_iter)
            # 记录结果
            # TODO：此处可以定义添加到日志项中的额外参数
            experiment.add_logMsg(...)
            # TODO：此处可以定义绘制历史趋势图时的绘图参数
            experiment.register_result(trainer.module, train_log, test_log, ...)
            del data_iter_generator, test_iter, trainer, train_log, test_log
