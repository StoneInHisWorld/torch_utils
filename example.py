import numpy as np
import torch

import utils.func.torch_tools
from data_related import data_related as dr
from networks.nets.wzynet_essay import WZYNetEssay
from utils.func import pytools
from data_related.datasets import DataSet
from utils.hypa_control import ControlPanel

# TODO: 选择选用的模型
Net = WZYNetEssay

# 调参面板
_ = None
# TODO： 请输入配置文件路径
cp = ControlPanel(_,  # 处理的数据集类
                  '',  # 调参json文件路径
                  '',  # 运行配置json文件路径
                  '',  # 结果存储文件路径
                  '')  # 训练成果网络存储路径
torch.random.manual_seed(cp.running_randomseed)

print('collecting data...')
# TODO: 读取数据集
# TODO：数据集需要提供格式为（特征集，标签集）的数据集
# TODO：自定义数据集需要切分为训练集、验证集、测试集
data = _
acc_func = _
dataset_name = _
train_data = _
valid_data = _
test_data = _

print('preprocessing...')
# TODO：对数据集进行预处理
# TODO：将数据集转化为DataSet对象
train_ds = DataSet(train_data[0], train_data[1])
valid_ds = DataSet(valid_data[0], valid_data[1])
test_ds = DataSet(test_data[0], test_data[1])
del data

features_preprocess = [
    # TODO：添加特征预处理函数
]
labels_process = [
    # TODO：添加标签预处理函数
]
train_ds.apply(features_preprocess, labels_process)
valid_ds.apply(features_preprocess, labels_process)
test_ds.apply(features_preprocess, labels_process)

# 多组参数训练流水线
for trainer in cp:
    with trainer as hps:
        # 读取训练超参数
        k, base, epochs, batch_size, ls_fn, lr, optim_str, w_decay, init_meth = hps
        device = cp.running_device
        exp_no = cp.running_expno
        # 数据迁移
        train_ds.to(device)
        valid_ds.to(device)
        test_ds.to(device)
        # TODO：将数据集转化为数据加载器
        # TODO：可选用本块的直接加载
        train_iter = dr.to_loader(train_data, batch_size)
        valid_iter = dr.to_loader(valid_data)
        test_iter = dr.to_loader(test_data)

        # TODO：可选用本块的k折加载
        # train_sampler_iter = dr.k_fold_split(train_ds, k=k)
        # train_loaders = (
        #     (dr.to_loader(train_ds, batch_size, sampler=train_sampler),
        #      dr.to_loader(train_ds, sampler=valid_sampler))
        #     for train_sampler, valid_sampler in train_sampler_iter
        # )  # 将抽取器遍历，构造加载器

        # TODO：可选用本块的懒加载

        print('constructing WZYNetwork...')
        # 构建网络
        # TODO：填充网络初始参数
        in_channel = _
        out_features = _
        net = Net(in_channel, base, out_features, init_meth=init_meth, device=device)
        cp.__list_net(net, (in_channel, *train_data.feature_shape), batch_size)

        print(f'training on {device}...')
        # 进行训练准备
        optimizer = utils.func.torch_tools.get_optimizer(net, optim_str, lr, w_decay)
        ls_fn = utils.func.torch_tools.get_ls_fn(ls_fn)
        history = net.train_(
            train_iter, valid_iter=valid_iter, optimizers=optimizer, num_epochs=epochs,
            ls_fn=ls_fn, criterion_a=acc_func
        )

        print('testing...')
        # TODO：训练历史参数提取
        train_acc, train_l = history["train_acc"][-1], history["train_l"][-1]
        try:
            valid_acc, valid_l = history["valid_acc"][-1], history["valid_l"][-1]
        except AttributeError as _:
            valid_acc, valid_l = np.nan, np.nan
        test_acc, test_ls = net.test_(test_iter, acc_func, ls_fn)
        print(f'train_acc = {train_acc * 100:.3f}%, train_l = {train_l:.5f}, '
              f'valid_acc = {valid_acc * 100:.3f}%, valid_l = {valid_l:.5f}, '
              f'test_acc = {test_acc * 100:.3f}%, test_l = {test_ls:.5f}')
        cp.__plot_history(
            history, xlabel='num_epochs', ylabel=f'loss({ls_fn.__class__.__name__})',
            title=f'dataset: {dataset_name} optimizer: {optimizer.__class__.__name__}\n'
                  f'net: {net.__class__.__name__}',
            save_path=None  # TODO：若要保存运行变化图，请指定本路径
        )
        # TODO：可选记录在日志中的数据列
        trainer.add_logMsg(
            True,
            train_l=train_l, train_acc=train_acc, valid_l=valid_l, valid_acc=valid_acc,
            test_acc=test_acc, test_ls=test_ls, exp_no=exp_no, dataset=dataset_name,
            random_seed=cp.running_randomseed, data_portion=cp.running_dataportion
        )
        # TODO：可选是否保存训练好的网络
        trainer.save_net(net, exp_no)
        del ls_fn, optimizer, history, net
