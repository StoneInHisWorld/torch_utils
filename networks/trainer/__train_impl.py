import time
from threading import Thread
from typing import Tuple

import torch
from tqdm import tqdm

from networks import BasicNN
from utils import History, Accumulator
from utils import ptools
from . import _prepare_train, _before_training, _after_training
from . import _prepare_valid, _get_a_progress_bar
from . import vduration_names, tduration_names
from .__hook_impl import hook
from .__log_impl import log_impl, log_summarize
from .__profiler_impl import profiling_impl

debug = False

def __add_lr_to_history(net, history):
    lr_names, lrs = net.get_lr_groups()
    history.add([f"{ln}_lrs" for ln in lr_names], lrs)


@_prepare_train
@hook()
def train_impl(trainer, train_iter) -> History:
    """简单训练实现。
    从train_iter中每次取出一批次数据进行前反向传播后计算评价指标，记录到History对象中。

    本函数的实际接口为
    def __train(
        module_class: type, __m_init_args: tuple, __m_init_kwargs: dict, prepare_args: tuple,
        train_iter: DataLoader
    ) -> History

    module_class、__m_init_args、__m_init_kwargs会被传输给net_builder作为神经网络创建参数，
    prepare_args用于给prepare装饰器进行网络训练准备

    :param train_iter: 训练数据迭代器
    :return: 训练数据记录`History`对象
    """
    raise NotImplementedError("暂时未测试过！")
    # 提取训练器参数
    pbar = trainer.pbar
    net = trainer.module
    criterion_a = trainer.criterion_a
    # n_epochs = trainer.hps['epochs']
    optimizer_s = net.optimizer_s
    scheduler_s = net.scheduler_s
    # 损失项
    loss_names = [f'train_{item}' for item in net.train_ls_names]
    # 评价项
    criteria_names = [f'train_{ptools.get_computer_name(criterion)}' for criterion in criterion_a]
    # 学习率项
    lr_names = [f'{lr}_lrs' for lr in net.lr_names]
    history = History(*(criteria_names + loss_names + lr_names))
    for epoch in range(trainer.n_epochs):
        pbar.set_description(f'世代{epoch + 1}/{trainer.n_epochs} 训练中……')
        __add_lr_to_history(net, history)
        # 记录批次训练损失总和，评价指标，样本数
        metric = Accumulator(len(loss_names + criteria_names) + 1)
        # 训练主循环
        for X, y in train_iter:
            net.train()
            preds, ls_es = net.forward_backward(X, y)
            # with torch.no_grad():
            #     num_examples = len(preds)
            #     correct_s = []
            #     for criterion in criterion_a:
            #         correct = criterion(preds, y)
            #         correct_s.append(correct)
            #     metric.add(
            #         *correct_s, *[ls * num_examples for ls in ls_es],
            #         num_examples
            #     )
            log_impl(preds, y, ls_es, criterion_a, metric)
            pbar.update(1)
        for scheduler in scheduler_s:
            scheduler.step()
        history.add(
            criteria_names + loss_names,
            [metric[i] / metric[-1] for i in range(len(metric) - 1)]
        )
    return history


@_prepare_train
@hook()
def train_and_valid_impl(trainer, train_iter, valid_iter):
    """训练和验证实现函数。
    从train_iter中每次取出一批次数据进行前反向传播后计算评价指标获得训练日志，随后调用__valid()函数进行验证获得验证日志，
    最后将两者记录到History对象中。

    本函数的实际签名为：
    def __train_and_valid(
        module_class: type, __m_init_args: tuple, __m_init_kwargs: dict, prepare_args: tuple,
        train_iter: DataLoader, valid_iter: DataLoader,
    ) -> History
    module_class、__m_init_args、__m_init_kwargs会被传输给net_builder作为神经网络创建参数，
    prepare_args用于给prepare装饰器进行网络训练准备

    :param train_iter: 训练数据供给迭代器
    :param valid_iter: 验证数据供给迭代器
    :return: 训练数据记录`History`对象
    """
    # 提取训练器参数
    net = trainer.module
    criterion_a = trainer.criterion_a
    # n_epochs = trainer.hps['epochs']
    # 损失项
    l_names = [f'train_{item}' for item in net.train_ls_names]
    # 评价项
    c_names = [f'train_{ptools.get_computer_name(criterion)}' for criterion in criterion_a]
    # 学习率项
    lr_names = [f'{lr}_lrs' for lr in net.lr_names]
    # duration_names = ["duration_train_data_fetch", "duration_for_back_ward", "duration_train_log"]
    duration_names = tduration_names
    # 记录对象
    metric_history = History(*(c_names + l_names + lr_names))
    duration_history = History(*duration_names)
    # 世代迭代主循环
    for epoch in range(trainer.n_epochs):
        trainer.pbar.set_description(f'世代{epoch + 1}/{trainer.n_epochs}训练中')
        __add_lr_to_history(net, metric_history)
        # 记录批次训练损失总和，评价指标，样本数
        metric_acc = Accumulator(len(l_names + c_names) + 1)
        duration_acc = Accumulator(len(duration_names) + 1)
        # 训练主循环
        logged_stamp = time.perf_counter()
        for fea_s, lbs in train_iter:
            data_fetched_stamp = time.perf_counter()
            # 前反向传播
            preds, ls_es = net.forward_backward(fea_s, lbs)
            fb_ward_stamp = time.perf_counter()
            durations = [
                data_fetched_stamp - logged_stamp,
                fb_ward_stamp - data_fetched_stamp,
            ]
            # 记录数据
            log_impl(
                trainer.pbar, preds, lbs, ls_es, l_names,
                criterion_a, c_names, metric_acc, durations, duration_acc
            )
            logged_stamp = time.perf_counter()
        # 进行学习率更新
        net.update_lr()
        vmetric_log, vduration_log = __valid(trainer, valid_iter, epoch)
        # 生成训练日志
        tmetric_log, tduration_log = log_summarize(
            metric_acc, duration_acc, c_names, l_names, duration_names
        )
        # 记录训练验证世代的时间、指标和损失值
        duration_history.add(
            [*tduration_log, *vduration_log, "duration_valid"],
            [*tduration_log.values(), *vduration_log.values(),
             time.perf_counter() - logged_stamp]
        )
        metric_history.add(
            [*tmetric_log, *vmetric_log],
            [*tmetric_log.values(), *vmetric_log.values()]
        )
        # duration_history.add(
        #     list(tduration_log.keys()) + list(vduration_log.keys()) +
        #     ["duration_valid"],
        #     list(tduration_log.values()) + list(vduration_log.values()) +
        #     [time.perf_counter() - logged_stamp]
        # )
        # metric_history.add(
        #     list(tmetric_log.keys()) + list(vmetric_log.keys()),
        #     list(tmetric_log.values()) + list(vmetric_log.values())
        # )
        # duration_history.add(
        #     duration_names + list(vduration_log.keys()) + ["duration_valid"],
        #     [duration_acc[i] / duration_acc[-1] for i in range(len(duration_acc) - 1)] +
        #     list(vduration_log.values()) + [time.perf_counter() - logged_stamp]
        # )
        # metric_history.add(
        #     c_names + l_names + list(vmetric_log.keys()),
        #     [metric_acc[i] / metric_acc[-1] for i in range(len(metric_acc) - 1)] +
        #     list(vmetric_log.values())
        # )
    return metric_history, duration_history


def train_with_k_fold(trainer, train_loaders_iter) -> History:
    """k-折训练实现

    拆解数据加载器供给器为k折，每折调用__train_and_valid()函数进行训练，获取训练日志后整合成k折训练日志。

    本函数的实际签名为：
    def __train_with_k_fold(
        module_class: type, __m_init_args: tuple, __m_init_kwargs: dict, prepare_args: tuple,
        train_loaders_iter: Generator[DataLoader]
    ) -> History

    module_class、__m_init_args、__m_init_kwargs会被传输给net_builder作为神经网络创建参数，
    prepare_args用于给prepare装饰器进行网络训练准备

    :param train_loaders_iter: 数据加载器供给器，提供k折验证的每一次训练所需训练集加载器、验证集加载器
    :return: k折训练日志
    """
    # # 创建网络
    # print(f'\r本次训练位于设备{trainer.runtime_cfg["device"]}上')
    # setattr(trainer, "module", trainer.net_builder.build(True))
    # # 设置进度条
    # pbar = get_a_training_progress_bar(trainer.n_epochs)
    # setattr(trainer, "pbar", pbar)
    raise NotImplementedError("暂时未测试过！")
    _before_training(trainer, trainer.n_epochs)
    pbar = trainer.pbar
    # 根据训练器参数调用相应的训练函数
    k_fold_history = None
    for i, (train_iter, valid_iter) in enumerate(train_loaders_iter):
        pbar.set_description(f'\r正在训练折{i + 1}')
        # 计算训练批次数
        pbar.total = trainer.k * trainer.n_epochs * (len(train_iter) + len(valid_iter))
        if ptools.is_multiprocessing(trainer.n_workers):
            raise NotImplementedError("暂未实现k_fold的多进程训练！")
            histories = tv_multiprocessing_impl(trainer, train_iter, valid_iter)
        else:
            histories = train_and_valid_impl(trainer, train_iter, valid_iter)
        k_fold_history = histories if k_fold_history is None else (
            k_fold_history[0] + histories[0], k_fold_history[1] + histories[1]
        )
        pbar.set_description(f'\r折{i + 1}训练完毕')
    # trainer.module.deactivate()
    # del trainer.pbar
    # 解除k_fold训练状态
    trainer.k = 1
    _after_training(trainer)
    return k_fold_history


@_prepare_valid
def __valid(trainer, valid_iter, epoch) -> Tuple[dict, dict]:
    """验证函数实现
    每次取出验证数据供给器中的下一批次数据进行前向传播，之后计算评价指标和损失，生成验证日志。

    :param valid_iter: 验证数据供给器
    :return: 验证记录
    """
    # 提取出验证所需参数
    criterion_a = trainer.criterion_a
    net = trainer.module
    # 要统计的数据种类数目
    l_names = [f'valid_{item}' for item in net.test_ls_names]
    # duration_names = ["duration_vdata_fetch", "duration_vpredict", "duration_vlog"]
    duration_names = vduration_names
    c_names = [f'valid_{ptools.get_computer_name(criterion)}' for criterion in criterion_a]
    # 记录对象
    metric_acc = Accumulator(len(criterion_a) + len(l_names) + 1)
    duration_acc = Accumulator(len(duration_names) + 1)
    # 计算准确率和损失值
    logged_stamp = time.perf_counter()
    # 开始对验证集中的数据进行验证
    for features, labels in valid_iter:
        data_fetched_stamp = time.perf_counter()
        # 预测
        preds, ls_es = net.forward_backward(features, labels)
        predicted_stamp = time.perf_counter()
        # 计算出数据获取时间和预测时间
        durations = [
            data_fetched_stamp - logged_stamp,
            predicted_stamp - data_fetched_stamp
        ]
        # 记录
        log_impl(
            trainer.pbar, preds, labels, ls_es, l_names, criterion_a, c_names,
            metric_acc, durations, duration_acc
        )
        logged_stamp = time.perf_counter()
    # 生成验证日志。迭代轮数大于1时，才生成记录日志。
    return log_summarize(metric_acc, duration_acc, 
                         c_names, l_names, duration_names)
    # if metric_acc[-1] > 0:
    #     metric_log = {
    #         name: metric_acc[i] / metric_acc[-1]
    #         for i, name in enumerate(c_names + l_names)
    #     }
    #     duration_log = {
    #         name: duration_acc[i] / duration_acc[-1]
    #         for i, name in enumerate(duration_names)
    #     }
    # else:
    #     metric_log, duration_log = {}, {}
    # i += 1
    # for j, n in enumerate(l_names + duration_names):
    #     metric_log[n] = metric_acc[i + j] / metric_acc[-1]
    # for
    # return metric_log, duration_log

#
# def __train_and_valid_with_preprocessing(self, train_iter, valid_iter) -> History:
#     """多进程训练实现
#     :param train_iter: 训练数据迭代器
#     :param valid_iter: 验证数据迭代器
#     :return: 训练历史记录
#     """
#     from __subprocess_impl import train_valid_impl
#     # 提取训练器参数
#     pbar = self.pbar
#     del self.pbar
#     n_epochs = self.hps['epochs']
#
#     pbar.set_description('\r正在创建队列和事件对象……')
#     # 进程通信队列
#     # TODO：改成双工Pipe实现
#     ctx = torch.multiprocessing.get_context("spawn")
#     tdata_q = ctx.Queue(int(self.runtime_cfg['train_prefetch']))  # 传递训练数据队列
#     vdata_q = ctx.Queue(int(self.runtime_cfg['valid_prefetch']))  # 传递验证数据队列
#     pbar_q = ctx.Queue()  # 传递进度条更新消息队列
#     epoch_q = ctx.Queue()  # 传递世代更新消息队列
#
#     def update_pbar():
#         msg = pbar_q.get()
#         while msg:
#             assert isinstance(msg, int) or isinstance(msg, str), "进度条更新只接受数字或字符串更新！"
#             if isinstance(msg, int):
#                 pbar.update(msg)
#             else:
#                 pbar.set_description(msg)
#             msg = pbar_q.get()
#
#     # 生成子进程用于创建网络、执行网络更新并记录数据
#     # 搭建输出结果通信管道
#     parent_conn, child_conn = ctx.Pipe(duplex=False)
#     parent_conn, child_conn = ctx.Pipe(duplex=False)
#     # 创建子线程进行训练和验证操作，并更新进度条
#     tv_subp = ctx.Process(target=train_valid_impl, args=(
#         self, tdata_q, vdata_q, pbar_q, epoch_q, child_conn
#     ))
#     pbar_update_thread = Thread(target=update_pbar)  # 更新进度条
#     # 开启两个子进程
#     pbar_update_thread.start()
#     tv_subp.start()
#     # 获取所有的数据，并且发送给训练进程
#     for epoch in range(n_epochs):
#         # 通知子进程新的世代开始了
#         epoch_q.put(epoch)
#         pbar.set_description(f'获取世代{epoch + 1}/{n_epochs}的训练数据……')
#         # 不断从训练数据集迭代器中取训练数据
#         for X, y in train_iter:
#             tdata_q.put((X, y))
#         # 通知训练进程，当前世代的数据已经传递完毕
#         tdata_q.put(None)
#         pbar.set_description(f'获取世代{epoch + 1}/{n_epochs}的验证数据……')
#         # 从验证迭代器中取验证数据
#         for X, y in valid_iter:
#             vdata_q.put((X, y))
#         # 通知验证进程，当前世代的数据已经传递完毕
#         vdata_q.put(None)
#         pbar.set_description(f'世代{epoch + 1}/{n_epochs} 数据获取完毕，等待网络消耗剩下的数据')
#     # 使用None通知子进程数据已经获取完毕
#     # tdata_q.put(None)
#     # vdata_q.put(None)
#     epoch_q.put(None)
#     # 处理随机顺序返回的结果
#     ret = [parent_conn.recv(), parent_conn.recv()]
#     if isinstance(ret[0], History) and isinstance(ret[1], BasicNN):
#         history, self.module = ret
#     elif isinstance(ret[0], BasicNN) and isinstance(ret[1], History):
#         self.module, history = ret
#     else:
#         raise ValueError(f"多进程管道接收到了异常的数据类型，为{type(ret[0])}和{type(ret[1])}")
#     return history

def tv_multiprocessing_impl(trainer, train_iter, valid_iter):
    """多进程训练实现
    :param train_iter: 训练数据迭代器
    :param valid_iter: 验证数据迭代器
    :return: 训练历史记录
    """
    from networks.trainer.__new_subprocess_impl import train_valid_impl

    # 提取训练器参数
    n_epochs = trainer.n_epochs
    pbar = _get_a_progress_bar(n_epochs * (len(train_iter) + len(valid_iter)), trainer.pbar_verbose)
    pbar.set_description('\r正在创建队列和事件对象')
    # 进程通信队列
    ctx = torch.multiprocessing.get_context("spawn")
    tdata_q = ctx.Queue(trainer.train_prefetch)  # 传递训练数据队列
    vdata_q = ctx.Queue(trainer.valid_prefetch)  # 传递验证数据队列
    pbar_q = ctx.Queue()  # 传递进度条更新消息队列
    epoch_q = ctx.Queue()  # 传递世代更新消息队列

    def update_pbar():
        msg = pbar_q.get()
        while msg is not None:
            if isinstance(msg, int):
                pbar.update(msg)
            elif isinstance(msg, str):
                pbar.set_description(msg)
            elif isinstance(msg, dict):
                pbar.set_postfix(**msg)
            else:
                raise ValueError(f"不支持的进度条更新消息{msg}，进度条更新只接受数字、字符串或词典更新！")
            msg = pbar_q.get()
            # print(msg)

    def send_data(data_iter, data_q, epoch, which):
        pbar.set_description(f'获取世代{epoch}/{n_epochs}的{which}数据')
        n_data = 0
        for X, y in data_iter:
            data_q.put((X, y))
            if debug:
                print(f"线程发送了世代{epoch}/{n_epochs}的第{n_data}批次的{which}数据")
            n_data += 1
        data_q.put(None)

    # 生成子进程用于创建网络、执行网络更新并记录数据
    # 搭建输出结果通信管道
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    # 创建子线程进行训练和验证操作，并更新进度条
    tv_subp = ctx.Process(target=train_valid_impl, args=(
        trainer, 
        tdata_q, vdata_q, pbar_q, epoch_q,
        ctx, trainer.tdata_q_len, trainer.vdata_q_len,
        child_conn
    ))
    pbar_update_thread = Thread(target=update_pbar)  # 更新进度条
    # 开启两个子进程
    pbar_update_thread.start()
    tv_subp.start()
    # 获取所有的数据，并且发送给训练进程
    for epoch in range(1, n_epochs + 1):
        # 通知子进程新的世代开始了
        epoch_q.put(epoch)
        # 不断从迭代器中取数据
        tsending = Thread(target=send_data, args=(train_iter, tdata_q, epoch, "训练"))
        tsending.start()
        vsending = Thread(target=send_data, args=(valid_iter, vdata_q, epoch, "验证"))
        vsending.start()
        # 等待数据发送完毕
        tsending.join()
        vsending.join()
        pbar.set_description(f'世代{epoch}/{n_epochs}数据获取完毕，等待网络消耗剩下的数据')
    # 使用None通知子进程数据已经获取完毕
    epoch_q.put(None)
    # 处理随机顺序返回的结果
    net_and_histories = []
    for _ in range(3):
        net_and_histories.append(parent_conn.recv())
        pbar.set_description(f"已接收训练结果{_ + 1}/3")
    # net_and_histories = [parent_conn.recv(), parent_conn.recv(), parent_conn.recv()]
    tv_subp.join()
    histories = []
    for net_or_history in net_and_histories:
        if isinstance(net_or_history, History):
            histories.append(net_or_history)
        elif isinstance(net_or_history, BasicNN):
            trainer.module = net_or_history
        else:
            raise ValueError(f"多进程管道接收到了异常的数据类型为{type(net_or_history)}")

    def priority(history):
        train_metric = sum(prop_name.startswith("train_") for prop_name in dir(history))
        valid_metric = sum(prop_name.startswith("valid_") for prop_name in dir(history))
        return train_metric * valid_metric

    histories = list(sorted(histories, key=priority, reverse=True))
    # if isinstance(net_and_histories[0], History) and isinstance(net_and_histories[1], BasicNN):
    #     history, trainer.module = net_and_histories
    # elif isinstance(net_and_histories[0], BasicNN) and isinstance(net_and_histories[1], History):
    #     trainer.module, history = net_and_histories
    # else:
    #     raise ValueError(f"多进程管道接收到了异常的数据类型，为{type(net_and_histories[0])}和{type(net_and_histories[1])}")
    pbar_update_thread.join()
    return histories

#
# def __pipe_train_and_valid_with_preprocessing(trainer, train_iter, valid_iter) -> History:
#     """多进程训练实现
#     :param train_iter: 训练数据迭代器
#     :param valid_iter: 验证数据迭代器
#     :return: 训练历史记录
#     """
#     from networks.trainer.__pipe_subprocess_impl import train_valid_impl
#
#     # 提取训练器参数
#     pbar = trainer.pbar
#     del trainer.pbar
#     n_epochs = trainer.hps['epochs']
#
#     pbar.bar_format = None  # 使用默认的进度条格式
#     pbar.set_description('\r正在创建队列和事件对象……')
#     # 进程通信队列
#     ctx = torch.multiprocessing.get_context("spawn")
#     tdata_pc, tdata_cc = ctx.Pipe(False)  # 传递训练数据队列
#     vdata_pc, vdata_cc = ctx.Pipe(False)  # 传递验证数据队列
#     pbar_q = ctx.Queue()  # 传递进度条更新消息队列，会用于记录进程间通信
#     epoch_pc, epoch_cc = ctx.Pipe(duplex=False)  # 传递世代更新消息队列
#     parent_conn, child_conn = ctx.Pipe(duplex=False)  # 搭建输出结果通信管道
#
#     def update_pbar():
#         msg = pbar_q.get()
#         while msg is not None:
#             assert isinstance(msg, int) or isinstance(msg, str), "进度条更新只接受数字或字符串更新！"
#             if isinstance(msg, int):
#                 pbar.update(msg)
#             else:
#                 pbar.set_description(msg)
#             msg = pbar_q.get()
#
#     def send_data(data_iter, data_q, epoch, which):
#         pbar.set_description(f'获取世代{epoch + 1}/{n_epochs}的{which}数据……')
#         for X, y in data_iter:
#             data_q.send((X, y))
#         data_q.send(None)
#
#     # 生成子进程用于创建网络、执行网络更新并记录数据
#     # 创建子线程进行训练和验证操作，并更新进度条
#     tv_subp = ctx.Process(target=train_valid_impl, args=(
#         trainer, tdata_pc, vdata_pc, pbar_q, epoch_pc, child_conn
#     ))
#     pbar_update_thread = Thread(target=update_pbar)  # 更新进度条
#     # 开启两个子进程
#     pbar_update_thread.start()
#     tv_subp.start()
#     # 获取所有的数据，并且发送给训练进程
#     for epoch in range(1, n_epochs + 1):
#         # 通知子进程新的世代开始了
#         epoch_cc.send(epoch)
#         # 不断从迭代器中取数据
#         tsending = Thread(target=send_data, args=(train_iter, tdata_cc, epoch, "训练"))
#         tsending.start()
#         vsending = Thread(target=send_data, args=(valid_iter, vdata_cc, epoch, "验证"))
#         vsending.start()
#         # 等待数据发送完毕
#         tsending.join()
#         vsending.join()
#         pbar.set_description(f'世代{epoch + 1}/{n_epochs} 数据获取完毕，等待网络消耗剩下的数据')
#     # 使用None通知子进程数据已经获取完毕
#     epoch_cc.send(None)
#     # 处理随机顺序返回的结果
#     ret = [parent_conn.recv(), parent_conn.recv()]
#     if isinstance(ret[0], History) and isinstance(ret[1], BasicNN):
#         history, trainer.module = ret
#     elif isinstance(ret[0], BasicNN) and isinstance(ret[1], History):
#         trainer.module, history = ret
#     else:
#         raise ValueError(f"多进程管道接收到了异常的数据类型，为{type(ret[0])}和{type(ret[1])}")
#     pbar_update_thread.join()
#     tv_subp.join()
#     return history

def train_with_profiler(trainer, data_iter, log_path):
    # 提取训练器参数
    raise NotImplementedError("暂时未测试过！")
    n_epochs = 2
    # 取相对较少数量的那个数据迭代器进行性能测试
    if trainer.k > 1:
        _, data_iter = next(data_iter)
    else:
        # 提取训练迭代器和验证迭代器
        data_iter = list(data_iter)
        if len(data_iter) == 2:
            _, data_iter = [it[0] for it in data_iter]
        elif len(data_iter) == 1:
            data_iter, _ = data_iter[0][0], None
        else:
            raise ValueError(f"无法识别的数据迭代器，其提供的长度为{len(data_iter)}")

    # 进度条设置
    trainer.pbar = tqdm(
        total=len(data_iter) * n_epochs, unit='批', position=0,
        desc=f'正在进行训练准备……', mininterval=1, ncols=100
    )
    profiling_impl(n_epochs, log_path, trainer, data_iter)

