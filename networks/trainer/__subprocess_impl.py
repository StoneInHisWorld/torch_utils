import traceback
import warnings

import dill
from torch.multiprocessing import SimpleQueue as PQueue, Event as PEvent

from utils.accumulator import Accumulator
from networks.decorators import prepare
from utils.func.pytools import get_computer_name
from utils.history import History


def data_iter_impl(
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
            # for i, batch in enumerate(data_iter):
            for batch in data_iter:
                data_Q.put(batch)
                # print(f'b{i}', end='')
                if end_env.is_set():
                    raise InterruptedError('获取数据时被中断！')
        # 放置None作为数据供给结束标志
        # print('data_fetching ends')
        data_Q.put(None)
        # print('data_iter_subpro_impl ends')
        end_env.wait()
    except Exception as e:
        traceback.print_exc()
        data_Q.put(e)
        # print('data_iter_subpro_impl ends')


def when_training_epoch_ends(net,
                             optimizer_s, scheduler_s,
                             log_Q, first_epoch=False):
    if first_epoch:
        # 如果是第一次世代更新，则只放入学习率组
        log_Q.put([
            [
                [param['lr'] for param in optimizer.param_groups]
                for optimizer in optimizer_s
            ],
        ])
    else:
        # log队列中放入每个优化器的学习率组以及网络参数
        log_Q.put([
            [
                [param['lr'] for param in optimizer.param_groups]
                for optimizer in optimizer_s
            ],
            {k: v.detach().clone() for k, v in net.state_dict().items()}
        ])
        # 更新学习率变化器组
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            for scheduler in scheduler_s:
                scheduler.step()


@prepare('train', True)
def train_impl(
        trainer,  # 训练器对象
        pbar_Q: PQueue, eval_Q: PQueue, log_Q: PQueue, data_Q: PQueue,  # 队列
        end_env: PEvent
):
    """
    进行data_Q.get()、pbar_Q.put()以及log_Q.put()
    :param net:
    :param optimizer_s:
    :param lr_scheduler_s:
    :param pbar_Q:
    :param log:
    :param data_Q:
    :param end_env:
    :return:
    """
    try:
        # 提取训练器参数
        net = trainer.module
        optimizer_s = net.optimizer_s
        scheduler_s = net.scheduler_s
        first_epoch = True
        while True:
            item = data_Q.get()
            if item is None:
                # 收到了None，认为是数据供给结束标志，将训练好的网路传递给记录进程进行验证和记录。
                when_training_epoch_ends(net, optimizer_s, scheduler_s, log_Q)
                break
            elif isinstance(item, Exception):
                # 收到了异常，则抛出
                raise item
            elif isinstance(item, str):
                # 收到了字符串，认为是世代开始标志
                pbar_Q.put(f'世代{item} 训练中……')
                when_training_epoch_ends(net, optimizer_s, scheduler_s, log_Q, first_epoch)
                eval_Q.put(first_epoch)
                first_epoch = False
            # 收到了元组，则认为是数据批次
            elif isinstance(item, tuple):
                X, y = item
                net.train()
                # 得到预测值以及损失组，即对应每个损失函数的损失值
                pred, ls_es = net.forward_backward(X, y)
                ls_es = [ls.detach() for ls in ls_es]
                # 将网络参数、预测值、损失值、标签值作为信号放入队列中
                eval_Q.put((pred.detach(), ls_es, y.detach().clone()))
            else:
                raise ValueError(f'不识别的信号{item}！')
        # 通知data_iter进程结束，通知log_process结束
        del data_Q
        eval_Q.put(None)
        log_Q.put(None)
        # 等待所有进程结束
        end_env.wait()
        return
    except Exception as e:
        traceback.print_exc()
        eval_Q.put(None)
        log_Q.put(None)


def eval_impl(
        loss_names, criterion_a, # 训练器对象
        pbar_Q: PQueue, eval_Q: PQueue, log_Q: PQueue,
        end_env: PEvent
):
    """在多线程处理中，训练以及验证数据的记录实现。
    进行log_Q.get()
    :param net: 网络结构
    :param criterion_a: 准确率计算函数
    :param eval_Q: 数据交换队列，train_impl通过该队列传输需要记录的信息
    :return: history历史记录
    """
    try:
        # 提取训练器参数
        # net = trainer.module
        # criterion_a = trainer.criterion_a
        # 损失项
        loss_names = [f'train_{item}' for item in loss_names]
        # 评价项
        criteria_names = [
            f'train_{get_computer_name(criterion)}' for criterion in criterion_a
        ]
        cri_los_names = criteria_names + loss_names
        metric = Accumulator(len(cri_los_names) + 1)

        while True:
            # 从队列中获取信号
            item = eval_Q.get()
            if item is None:
                # 收到了None，则认为是训练完成信号
                break
            elif isinstance(item, Exception):
                # 将异常抛出
                raise item
            elif isinstance(item, bool):
                # 如果收到了布尔值，则认为是世代开始标志
                if not item:
                    # 如果不是第一个世代开始，则将世代统计数据发送给日志进程
                    log_Q.put(metric)
                metric.reset()
            elif isinstance(item, tuple):
                # 如果收到了元组，则认为是迭代结束标志
                # 分解成预测值、损失值、标签值，进行评价指标的计算以及损失值的累加
                pred, ls_es, y = item
                num_examples = y.shape[0]
                metric.add(
                    *[criterion(pred, y) for criterion in criterion_a],  # 评价指标
                    *[ls * num_examples for ls in ls_es],  # 损失项
                    num_examples
                )
                del pred, ls_es, y
                # 完成了一批数据的计算，更新进度条
                pbar_Q.put(1)
            else:
                raise NotImplementedError(f'无法识别的信号{item}！')
            del item
        # 数据接收结束，向世代结束进程传递结束信号
        del eval_Q
        end_env.wait()
    except Exception as e:
        traceback.print_exc()
        pbar_Q.put(e)
        # print('tv_log_subprocess_impl ends')


def __train_logging(cri_los_names, history, train_metric):
    """进行训练参数的记录"""
    history.add(
        cri_los_names,
        [train_metric[i] / train_metric[-1] for i in range(len(train_metric) - 1)]
    )
    del train_metric


def tlog_impl(
        trainer,  # 训练器对象
        pbar_Q, log_Q,
        end_env
):
    """
    进行pbar_Q.put()
    :param net:
    :param valid_iter:
    :param pbar_Q:
    :return:
    """
    try:
        # 提取训练器参数
        net = trainer.module
        criterion_a = trainer.criterion_a
        # 学习率项
        lr_names = net.lr_names
        # 损失项
        loss_names = [f'train_{item}' for item in net.train_ls_names]
        # 评价项
        criteria_names = [
            f'train_{get_computer_name(criterion)}'
            for criterion in criterion_a
        ]
        cri_los_names = criteria_names + loss_names
        # 创建历史记录对象
        history = History(*(criteria_names + loss_names + lr_names))
        # 获取记录队列的消息
        while True:
            item = log_Q.get()
            if item is None:
                # 收到了None，则认为是训练完成信号
                break
            elif isinstance(item, Exception):
                # 将异常抛出
                raise item
            elif isinstance(item, list):
                lr_group = item[0]
                history.add(lr_names, lr_group)
                del lr_group
            elif isinstance(item, Accumulator):
                __train_logging(cri_los_names, history, item[0])
            else:
                raise NotImplementedError(f'无法识别的信号{item}！')
            del item
        del log_Q
        pbar_Q.put(history)
        end_env.set()
    except Exception as e:
        traceback.print_exc()
        pbar_Q.put(e)


@prepare("train", True)
@prepare("valid")
def vlog_impl(
        trainer,  # 训练器对象
        valid_iter,  # 数据加载器对象
        pbar_Q, log_Q,  # 队列对象
        end_env,  # 事件对象
):
    """
    进行pbar_Q.put()
    :param net:
    :param valid_iter:
    :param pbar_Q:
    :return:
    """
    try:
        # 提取训练器参数
        net = trainer.module
        criterion_a = trainer.criterion_a
        # 学习率项
        lr_names = net.lr_names
        # 损失项
        loss_names = [f'train_{item}' for item in net.train_ls_names]
        # 评价项
        criteria_names = [
            f'train_{get_computer_name(criterion)}' for criterion in criterion_a
        ]
        cri_los_names = criteria_names + loss_names
        # 创建历史记录对象
        history = History(*(criteria_names + loss_names + lr_names))
        # 对象化数据迭代器
        valid_iter = dill.loads(valid_iter)

        def _impl(item):
            """验证实现"""
            pbar_Q.put('正在加载验证模型……')
            state_dict = item[0]
            net.load_state_dict(state_dict)
            # 验证损失项
            tls_names = net.test_ls_names
            metric = Accumulator(len(criterion_a) + len(tls_names) + 1)
            # 计算准确率和损失值
            pbar_Q.put('验证中……')
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
            for j, loss_name in enumerate(tls_names):
                log['valid_' + loss_name] = metric[i + j] / metric[-1]
            history.add(list(log.keys()), list(log.values()))

        # 接受日志队列的消息
        while True:
            item = log_Q.get()
            if item is None:
                # 收到了None，则认为是训练完成信号
                break
            elif isinstance(item, Exception):
                # 将异常抛出
                raise item
            elif isinstance(item, list):
                lr_group, *item = item
                if len(item) > 0:
                    _impl(item)
                history.add(lr_names, lr_group)
                del lr_group
            elif isinstance(item, Accumulator):
                __train_logging(cri_los_names, history, item)
            else:
                raise NotImplementedError(f'无法识别的信号{item}！')
            del item
        del log_Q
        pbar_Q.put(history)
        end_env.set()
    except Exception as e:
        traceback.print_exc()
        pbar_Q.put(e)


# @prepare("valid")
# def tv_log_subprocess_impl(
#         trainer,  # 训练器对象
#         # net,  # 网络对象
#         # criterion_a,
#         valid_iter,  # 验证数据迭代器
#         log_Q: PQueue, pbar_Q: PQueue, end_env: PEvent  # 信号量相关
# ):
#     """在多线程处理中，训练以及验证数据的记录实现。
#     进行log_Q.get()
#     :param net: 网络结构
#     :param valid_iter: 验证数据迭代器
#     :param criterion_a: 准确率计算函数
#     :param log_Q: 数据交换队列，train_impl通过该队列传输需要记录的信息
#     :return: history历史记录
#     """
#     try:
#         # 提取训练器参数
#         net = trainer.module
#         criterion_a = trainer.criterion_a
#         # 学习率项
#         lr_names = net.lr_names
#         # 损失项
#         loss_names = [f'train_{item}' for item in net.loss_names]
#         # 评价项
#         criteria_names = [
#             f'train_{get_computer_name(criterion)}' for criterion in criterion_a
#         ]
#         cri_los_names = criteria_names + loss_names
#         # 设置历史记录对象
#         history = History(*(criteria_names + loss_names + lr_names))
#         # 创建训练模型的副本
#         valid_iter = dill.loads(valid_iter)
#         metric = Accumulator(len(cri_los_names) + 1)
#         while True:
#             # 从队列中获取信号
#             # item = log_Q.get(True)
#             item = log_Q.get()
#             # 收到了None，则认为是训练完成信号
#             if item is None:
#                 break
#             # 将异常抛出
#             elif isinstance(item, Exception):
#                 raise item
#             # 如果收到了序列，则认为是世代结束标志
#             elif isinstance(item, list):
#                 if len(item) == 2:
#                     # 若是获取到了学习率组和网络参数，则认为完成了一个世代的迭代，刷新累加器并进行验证
#                     lr_group, state_dict = item
#                     # 进行验证
#                     net.load_state_dict(state_dict)
#                     valid_log = valid_subprocess_impl(net, valid_iter, criterion_a, pbar_Q)
#                     # 记录训练和验证数据
#                     history.add(
#                         cri_los_names + list(valid_log.keys()),
#                         [metric[i] / metric[-1] for i in range(len(metric) - 1)] +
#                         list(valid_log.values())
#                     )
#                     metric.reset()
#                     del state_dict
#                 else:
#                     # 只收集到了学习率组
#                     [lr_group] = item
#                 # 记录学习率
#                 history.add(lr_names, lr_group)
#                 del lr_group
#             # 如果收到了元组，则认为是迭代结束标志
#             elif isinstance(item, tuple):
#                 # 分解成预测值、损失值、标签值，进行评价指标的计算以及损失值的累加
#                 pred, ls_es, y = item
#                 # 计算训练准确率数据
#                 correct_s = []
#                 for criterion in criterion_a:
#                     correct = criterion(pred, y)
#                     correct_s.append(correct)
#                 num_examples = y.shape[0]
#                 metric.add(
#                     *correct_s, *[ls * num_examples for ls in ls_es],
#                     num_examples
#                 )
#                 del pred, ls_es, y
#             else:
#                 raise NotImplementedError(f'无法识别的信号{item}！')
#             del item
#         # 将结果放在进度条队列中
#         # print(f'日志队列消耗完毕：{log_Q.empty()}')
#         pbar_Q.put(history)
#         # print('tv_log_subprocess_impl ends')
#         end_env.set()
#         return
#     except Exception as e:
#         traceback.print_exc()
#         pbar_Q.put(e)
#         # print('tv_log_subprocess_impl ends')
#
#
# @torch.no_grad()
# def valid_subprocess_impl(
#     net, valid_iter, criterion_a, pbar_Q
# ):
#     """
#     进行pbar_Q.put()
#     :param net:
#     :param valid_iter:
#     :param criterion_a:
#     :param pbar_Q:
#     :return:
#     """
#     net.eval()
#     # print('进行了验证')
#     pbar_Q.put('验证中……')
#     # 要统计的数据种类数目
#     criterion_a = criterion_a if isinstance(criterion_a, list) else [criterion_a]
#     # 要统计的数据种类数目
#     l_names = net.test_ls_names
#     metric = Accumulator(len(criterion_a) + len(l_names) + 1)
#     # 计算准确率和损失值
#     # for i, (features, labels) in enumerate(valid_iter):
#     for features, labels in valid_iter:
#         preds, ls_es = net.forward_backward(features, labels, False)
#         metric.add(
#             *[criterion(preds, labels) for criterion in criterion_a],
#             *[ls * len(features) for ls in ls_es],
#             len(features)
#         )
#         pbar_Q.put(1)
#         # print(f'v{i}', end=' ')
#     # 生成测试日志
#     log = {}
#     i = 0
#     for i, computer in enumerate(criterion_a):
#         try:
#             log['valid_' + get_computer_name(computer)] = metric[i] / metric[-1]
#         except AttributeError:
#             log['valid_' + get_computer_name(computer)] = metric[i] / metric[-1]
#     i += 1
#     for j, loss_name in enumerate(l_names):
#         log['valid_' + loss_name] = metric[i + j] / metric[-1]
#     return log
