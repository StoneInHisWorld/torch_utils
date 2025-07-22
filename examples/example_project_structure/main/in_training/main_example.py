import sys

sys.path.append('where/you/put/torch_utils')
sys.path.append('where/you/put/project')

from networks.trainer import Trainer
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
    # TODO: 请在这里实现数据迁移功能
    return batch


if __name__ == '__main__':
    # 调参面板
    cp = ControlPanel(DataSet, Net, '../../config')

    # 多组参数训练流水线
    for experiment in cp:
        with experiment as hps:
            print('正在整理数据……')
            data = DataSet(Net, cp["ds_kwargs"])
            criterion_a = DataSet.get_criterion_a()

            print('数据预处理中……')
            data_iter_generator, test_iter = data.to_dataloaders(
                hps['k'], hps['batch_size'], transit_fn=transit_fn, **cp['dl_kwargs']
            )
            fea_chan, fea_shape, lb_chan, lb_shape = (data.f_channel, data.f_req_shp,
                                                      data.l_channel, data.l_req_shp)
            del data

            # TODO: 设置网络构建参数
            net_init_kwargs = {...}
            net_init_args = (
                (hps['version'], 1, 1, hps['base']), {},
                (2, 64), {"norm_type": hps['norm_type']}
            )
            # TODO: 填写训练准备参数
            prepare_args = (
                [(hps['optim_str'], {'lr': hps['lr'], 'w_decay': hps['w_decay']}),
                 (hps['optim_str'], {'lr': hps['lr'], 'w_decay': hps['w_decay']})],
                [('step', {'step_size': hps['step_size'], 'gamma': hps['gamma']}),
                 ('step', {'step_size': hps['step_size'], 'gamma': hps['gamma']})],
                [(hps['ls_fn'], {'lambda_l1': hps['lambda_l1']})]
            )

            trainer = Trainer(
                Net, net_init_args, net_init_kwargs, (fea_chan, *fea_shape),
                prepare_args, criterion_a, cp['t_kwargs'], hps
            )
            # 开始训练、验证和测试
            train_log = trainer.train(data_iter_generator)
            test_log = trainer.test(test_iter)
            # 记录结果
            experiment.register_result(
                trainer.module, train_log, test_log,
                figsize=(20, 15), max_nrows=4
            )
            del data_iter_generator, test_iter, trainer, train_log, test_log
