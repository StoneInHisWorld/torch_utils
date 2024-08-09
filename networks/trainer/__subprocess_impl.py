import traceback
import warnings
from torch.multiprocessing import SimpleQueue as PQueue, Event as PEvent

import dill
import torch

from utils.accumulator import Accumulator
from utils.decorators import prepare
from utils.func.pytools import get_computer_name
from utils.history import History


def data_iter_subpro_impl(
        n_epochs, data_iter, data_Q: PQueue, end_env: PEvent
):
    """
    进行data_Q.put()
    :param n_epochs:
    :param data_iter:
    :param data_Q: 数据供给队列，需要给定长度以节省内存
    :return:
    """
    try:
        data_iter = dill.loads(data_iter)
        # 训练主循环
        for epoch in range(n_epochs):
            data_Q.put(f'{epoch + 1}/{n_epochs}')
            for batch in data_iter:
                data_Q.put(batch)
                if end_env.is_set():
                    raise InterruptedError('获取数据时被中断！')
            # 放置None作为世代结束标志
        # print('data_fetching ends')
        data_Q.put(None)
        # print('data_iter_subpro_impl ends')
        end_env.wait()
    except Exception as e:
        traceback.print_exc()
        data_Q.put(e)
        # print('data_iter_subpro_impl ends')


@prepare('train')
def train_subprocess_impl(
        trainer,  # 训练器对象
        # net,  # 网络对象
        pbar_Q: PQueue, log_Q: PQueue, data_Q: PQueue,  # 队列
        data_end_env: PEvent
):
    """
    进行data_Q.get()、pbar_Q.put()以及log_Q.put()
    :param net:
    :param optimizer_s:
    :param lr_scheduler_s:
    :param pbar_Q:
    :param log_Q:
    :param data_Q:
    :param data_end_env:
    :return:
    """
    try:
        # 提取训练器参数
        net = trainer.module
        optimizer_s = net.optimizer_s
        scheduler_s = net.scheduler_s
        while True:
            item = data_Q.get()
            # 收到了None，认为是数据供给结束标志
            if item is None:
                break
            # 收到了异常，则抛出
            elif isinstance(item, Exception):
                raise item
            # 收到了字符串，认为是世代结束标志
            elif isinstance(item, str):
                pbar_Q.put(f'世代{item} 训练中……')
                if item.startswith('1'):
                    # 如果是第一次世代更新，则只放入学习率组
                    log_Q.put([
                        [
                            [param['lr'] for param in optimizer.param_groups]
                            for optimizer in optimizer_s
                        ],
                        # ], True)
                    ])
                else:
                    # log队列中放入每个优化器的学习率组以及网络参数
                    log_Q.put([
                        [
                            [param['lr'] for param in optimizer.param_groups]
                            for optimizer in optimizer_s
                        ],
                        {k: v.detach().clone() for k, v in net.state_dict().items()}
                        # ], True)
                    ])
                    # 更新学习率变化器组
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=UserWarning)
                        for scheduler in scheduler_s:
                            scheduler.step()
            # 收到了元组，则认为是数据批次
            elif isinstance(item, tuple):
                X, y = item
                net.train()
                # 得到预测值以及损失组，即对应每个损失函数的损失值
                pred, ls_es = net.forward_backward(X, y)
                ls_es = [ls.detach() for ls in ls_es]
                # 将网络参数、预测值、损失值、标签值作为信号放入队列中
                # log_Q.put((pred.detach(), ls_es, y.detach().clone()), True)
                log_Q.put((pred.detach(), ls_es, y.detach().clone()))
                # 完成了一批数据的计算，更新进度条
                pbar_Q.put(1)
            else:
                raise ValueError(f'不识别的信号{item}！')
        # 通知data_iter进程结束，通知log_process结束
        # data_end_env.set()
        # log_Q.put(None, True)
        log_Q.put(None)
        # 等待记录进程结束
        # log_end_env.wait()
        # print('train_subpro_impl ends')
        data_end_env.wait()
        return
    except Exception as e:
        traceback.print_exc()
        log_Q.put(e)
        # print('train_subpro_impl ends')


@prepare("valid")
def tv_log_subprocess_impl(
        trainer,  # 训练器对象
        # net,  # 网络对象
        # criterion_a,
        valid_iter,  # 验证数据迭代器
        log_Q: PQueue, pbar_Q: PQueue, end_env: PEvent  # 信号量相关
):
    """在多线程处理中，训练以及验证数据的记录实现。
    进行log_Q.get()
    :param net: 网络结构
    :param valid_iter: 验证数据迭代器
    :param criterion_a: 准确率计算函数
    :param log_Q: 数据交换队列，train_impl通过该队列传输需要记录的信息
    :return: history历史记录
    """
    try:
        # 提取训练器参数
        net = trainer.module
        criterion_a = trainer.criterion_a
        # 学习率项
        lr_names = net.lr_names
        # 损失项
        loss_names = [f'train_{item}' for item in net.loss_names]
        # 评价项
        criteria_names = [
            f'train_{get_computer_name(criterion)}' for criterion in criterion_a
        ]
        cri_los_names = criteria_names + loss_names
        # 设置历史记录对象
        history = History(*(criteria_names + loss_names + lr_names))
        # 创建训练模型的副本
        valid_iter = dill.loads(valid_iter)
        metric = Accumulator(len(cri_los_names) + 1)
        while True:
            # 从队列中获取信号
            # item = log_Q.get(True)
            item = log_Q.get()
            # 收到了None，则认为是训练完成信号
            if item is None:
                break
            # 将异常抛出
            elif isinstance(item, Exception):
                raise item
            # 如果收到了序列，则认为是世代结束标志
            elif isinstance(item, list):
                if len(item) == 2:
                    # 若是获取到了学习率组和网络参数，则认为完成了一个世代的迭代，刷新累加器并进行验证
                    lr_group, state_dict = item
                    # 进行验证
                    net.load_state_dict(state_dict)
                    valid_log = valid_subprocess_impl(net, valid_iter, criterion_a, pbar_Q)
                    # 记录训练和验证数据
                    history.add(
                        cri_los_names + list(valid_log.keys()),
                        [metric[i] / metric[-1] for i in range(len(metric) - 1)] +
                        list(valid_log.values())
                    )
                    metric.reset()
                    del state_dict
                else:
                    [lr_group] = item
                # 记录学习率
                history.add(lr_names, lr_group)
                del lr_group
            # 如果收到了元组，则认为是迭代结束标志
            elif isinstance(item, tuple):
                # 分解成预测值、损失值、标签值，进行评价指标的计算以及损失值的累加
                pred, ls_es, y = item
                # 计算训练准确率数据
                correct_s = []
                for criterion in criterion_a:
                    correct = criterion(pred, y)
                    correct_s.append(correct)
                num_examples = y.shape[0]
                metric.add(
                    *correct_s, *[ls * num_examples for ls in ls_es],
                    num_examples
                )
                del pred, ls_es, y
            else:
                raise NotImplementedError(f'无法识别的信号{item}！')
            del item
        # 将结果放在进度条队列中
        pbar_Q.put(history)
        # print('tv_log_subprocess_impl ends')
        end_env.set()
        return
    except Exception as e:
        traceback.print_exc()
        pbar_Q.put(e)
        # print('tv_log_subprocess_impl ends')


@torch.no_grad()
def valid_subprocess_impl(
        net, valid_iter, criterion_a, pbar_Q
):
    """
    进行pbar_Q.put()
    :param net:
    :param valid_iter:
    :param criterion_a:
    :param pbar_Q:
    :return:
    """
    net.eval()
    pbar_Q.put('验证中……')
    # 要统计的数据种类数目
    criterion_a = criterion_a if isinstance(criterion_a, list) else [criterion_a]
    # 要统计的数据种类数目
    l_names = net.test_ls_names
    metric = Accumulator(len(criterion_a) + len(l_names) + 1)
    # 计算准确率和损失值
    for features, labels in valid_iter:
        preds, ls_es = net.forward_backward(features, labels, False)
        metric.add(
            *[criterion(preds, labels) for criterion in criterion_a],
            *[ls * len(features) for ls in ls_es],
            len(features)
        )
        pbar_Q.put(1)
    # 生成测试日志
    log = {}
    i = 0
    for i, computer in enumerate(criterion_a):
        try:
            log['valid_' + get_computer_name(computer)] = metric[i] / metric[-1]
        except AttributeError:
            log['valid_' + get_computer_name(computer)] = metric[i] / metric[-1]
    i += 1
    for j, loss_name in enumerate(l_names):
        log['valid_' + loss_name] = metric[i + j] / metric[-1]
    return log
