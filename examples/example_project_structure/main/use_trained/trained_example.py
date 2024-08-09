import utils.func.log_tools as ltools
from data_related import ds_operation as dso
from networks.trainer import Trainer
# TODO: 选择读取数据集的类
from your_DAO_module import your_DAO_class as DataSet
# TODO: 选择训练的网络架构类型
from your_net_module import your_Net as Net

"""此文件用于使用训练完成的网络模型可视化输出结果"""


def transit_fn(batch, **kwargs):
    """可自定义的数据供给传输函数
    DataLoader每次从内存取出数据后，都会调用本函数对批次进行预处理

    :param batch: 需要预处理的数据批
    :param kwargs: 预处理所用关键字参数
    :return: 数据批次
    """
    ...


net_name = Net.__name__.lower()
# TODO：指定需要查看的实验结果序号
read_queue = [...]
# TODO：指定训练负责的设备
device = 'your_device'
# TODO：指定日志存放路径
log_root = f'where/you/put/your/log'
log_path = f'the/name/of/your/log'

print('正在整理数据……')
# TODO: 设置数据集创建参数
which_dataset = 'choose which dataset under your data root'
data = DataSet('where/you/put/your/dataset', which_dataset, Net, 0.5, ...)
criterion_a = DataSet.get_criterion_a()

print('数据预处理中……')
_, test_ds = data.to_dataset()
# TODO：在此处定义数据供给器参数
data_iter = dso.to_loader(test_ds, transit_fn, ...)

for exp_no in read_queue:
    hp = ltools.get_logData(log_path, exp_no)
    print(f'---------------------------实验{exp_no}号的结果'
          f'---------------------------')
    # TODO：设置网络构建参数，按照网络的初始化参数进行指定
    net_init_args = (...,)
    net_init_kwargs = {...}
    # TODO：在此处设置网络预测参数。训练位置参数有三个，按位序为优化器参数、学习率规划器参数、损失函数参数。
    # 预测操作只需要损失函数参数，其要求为二元组列表，二元组内容为（损失函数类型字符串，损失函数构造关键字参数）
    prepare_args = ([], [], [...])
    trainer = Trainer(
        Net, net_init_args, net_init_kwargs, prepare_args,
        DataSet
    )
    results = trainer.predict(data_iter, DataSet.wrap_fn)
    # TODO：指定结果保存路径
    DataSet.save_fn(results, 'where/you/wanna/put/your/results')
    print(f'----------------------已保存实验{exp_no}号的结果'
          f'----------------------')
