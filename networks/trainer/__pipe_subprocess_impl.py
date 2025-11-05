import torch
from torch.multiprocessing import Process
from torch.multiprocessing import Queue

from networks.decorators import prepare
from utils import ptools, History, Thread
from utils.accumulator import Accumulator
from .__hook_impl import hook


@prepare('train')
@hook()
def train_valid_impl(trainer, tdata_pc, vdata_pc, pbar_q, epoch_pc, result_conn):
    net = trainer.module
    # 传递训练数据的管道
    ctx = torch.multiprocessing.get_context("spawn")
    tlog_pc, tlog_cc = ctx.Pipe(False)  # 训练数据
    vlog_pc, vlog_cc = ctx.Pipe(False)  # 验证数据
    lr_pc, lr_cc = ctx.Pipe(False)  # 学习率
    log_epoch_pc, log_epoch_cc = ctx.Pipe(False)  # 世代更新队列
    # 创建记录进程，处理评价指标的计算以及记录，损失值、学习率的记录的事项
    log_subp = Process(target=log_impl,
        args=(trainer.criterion_a, net.train_ls_names, net.test_ls_names, net.lr_names,
              lr_pc, tlog_pc, vlog_pc, log_epoch_pc, pbar_q, result_conn)
    )
    log_subp.start()
    # 通过队列获取世代更新消息。这里不能写成while q.get(): 因为这里的世代数会等于0
    epoch = epoch_pc.recv()
    while epoch is not None:
        batch = tdata_pc.recv()
        pbar_q.put(f"世代{epoch}训练开始")
        log_epoch_cc.send(epoch)
        # n_batch = 0
        # print(f"拿到世代{epoch}的{n_batch}批次的训练数据")
        while batch is not None:
            """"训练实现"""
            # 拿到批量数据
            X, y = batch
            # print(f"世代{epoch}的{n_batch}批次开始前反向传播")
            # 前反向传播
            pred_s, ls_es = net.forward_backward(X, y)
            # print(f"世代{epoch}的{n_batch}批次前反向传播已经完成")
            # 数据传递给记录进程，进行评价指标计算、历史记录更新（损失值、评价指标、学习率）
            log_data = pred_s.detach().clone(), y.detach().clone(), [l.detach().clone() for l in ls_es]
            tlog_cc.send(log_data)
            # print(f"世代{epoch}的{n_batch}批次训练结果已经发送")
            # 获取下一批量数据
            batch = tdata_pc.recv()
            # n_batch += 1
            # if batch is not None:
            #     print(f"拿到世代{epoch}的{n_batch}批次的训练数据")
        lr_cc.send(net.get_lr_groups())
        # print(f"世代{epoch}的学习率组已经发送")
        # 更新优化器学习率
        net.update_lr()
        tlog_cc.send(None)
        pbar_q.put(f"世代{epoch}训练完毕")
        # print(f"世代{epoch}训练完毕")
        # 验证
        __valid_impl(trainer, vdata_pc, vlog_cc, pbar_q, epoch)
        epoch = epoch_pc.recv()
    # 记录最后一次学习率，并通知学习率已经记录完毕
    log_epoch_cc.send(None)
    lr_cc.send(None)
    # 使用None来通知记录进程“训练已经结束”
    tlog_cc.send(None)
    pbar_q.put(None)
    result_conn.send(net)
    log_subp.join()


@torch.no_grad()
@prepare('valid')
def __valid_impl(trainer, vdata_pc, vlog_cc, pbar_q, epoch):
    """验证函数实现
    每次取出验证数据供给器中的下一批次数据进行前向传播，之后计算评价指标和损失，生成验证日志。

    :param valid_iter: 验证数据供给器
    :return: 验证记录
    """
    # 提取出验证所需参数
    net = trainer.module
    # 固定网络
    batch = vdata_pc.recv()
    # n_batch = 0
    # print(f"拿到世代{epoch}的{n_batch}批次的验证数据")
    pbar_q.put(f"世代{epoch}验证开始")
    while batch is not None:
        X, y = batch  # epoch应该每次都相同
        pred_s, ls_es = net.forward_backward(X, y, False)
        # 数据传递给记录进程，进行评价指标计算、历史记录更新（损失值、评价指标、学习率）
        log_data = pred_s.detach().clone(), y.detach().clone(), [l.detach().clone() for l in ls_es]
        vlog_cc.send(log_data)
        # print(f"世代{epoch}的{n_batch}批次的验证数据已经发送")
        # 获取下一批数据
        batch = vdata_pc.recv()
        # n_batch += 1
        # print(f"拿到世代{epoch}的{n_batch}批次的验证数据")
    vlog_cc.send(None)
    pbar_q.put(f"世代{epoch}验证完毕")
    # print(f"世代{epoch}验证完毕")


def log_impl(criteria_fns, trls_names, tels_names, lr_names,
             lrlog_pc, tlog_pc, vlog_pc, epoch_pc, pbar_q, result_conn):
    """管理整个训练过程的历史记录"""
    # 评价项名称
    tcf_names = [f'train_{ptools.get_computer_name(cfn)}' for cfn in criteria_fns]
    tls_names = [f'train_{ln}' for ln in trls_names]
    vcf_names = [f'valid_{ptools.get_computer_name(cfn)}' for cfn in criteria_fns]
    vls_names = [f'valid_{ln}' for ln in tels_names]
    lr_names = [f'{lr}_lrs' for lr in lr_names]
    # 创建历史记录
    history_keys = tcf_names + tls_names + vcf_names + vls_names + lr_names
    history = History(*history_keys)

    # 创建线程进行学习率的记录
    def add_lr_to_history():
        data = lrlog_pc.recv()
        while data:
            lr_names, lrs = data
            history.add([f"{ln}_lrs" for ln in lr_names], lrs)
            data = lrlog_pc.recv()

    lrlog_thread = Thread(add_lr_to_history)
    lrlog_thread.start()
    # 每次世代开始就创建训练数据记录线程和验证数据记录线程，得到其结果后记录到历史记录对象中
    tmetric_acc = Accumulator(len(trls_names + criteria_fns) + 1)
    vmetric_acc = Accumulator(len(trls_names + criteria_fns) + 1)
    # 接收世代更新消息。这里不能写成while q.get(): 因为这里的世代数会等于0
    epoch = epoch_pc.recv()
    # print(f"开始记录世代{epoch}")
    while epoch is not None:
        # 创建训练数据记录线程
        tmetric_acc.reset()
        tlog_thread = Thread(__log_impl, epoch, "训练", tlog_pc, criteria_fns, tmetric_acc, pbar_q)
        tlog_thread.start()
        # 创建验证数据记录线程
        vmetric_acc.reset()
        vlog_thread = Thread(__log_impl, epoch, "验证", vlog_pc, criteria_fns, vmetric_acc, pbar_q)
        vlog_thread.start()
        # 等待数据记录完毕
        tlog_thread.join()
        # print(f"世代{epoch}训练数据计算完毕")
        vlog_thread.join()
        # print(f"世代{epoch}验证数据计算完毕")
        # 统计验证数据并加入到历史记录中
        history.add(
            list(filter(lambda k: "_lrs" not in k, history_keys)), [
                *[tmetric_acc[i] / tmetric_acc[-1] for i in range(len(tmetric_acc) - 1)],
                *[vmetric_acc[i] / vmetric_acc[-1] for i in range(len(vmetric_acc) - 1)]
            ]
        )
        pbar_q.put(f"世代{epoch}记录完毕")
        epoch = epoch_pc.recv()
    pbar_q.put(None)
    result_conn.send(history)
    result_conn.close()

@torch.no_grad()
def __log_impl(epoch, which, log_pc, criteria_fns, metric_acc, pbar_q):
    """管理训练过程单个世代
    :param metric_acc: 主进程提供的数据累加器，只用于累加训练指标、损失值和样本数
    """
    # n_data = 0
    data = log_pc.recv()
    # print(f"线程拿到了第{epoch}世代的{n_data}批次的{which}数据")
    while data is not None:
        """训练数据记录"""
        pred_s, y, ls_es = data
        num_examples = len(pred_s)
        # 记录和计算指标，记录损失
        metric_acc.add(
            *[cfn(pred_s, y) for cfn in criteria_fns],
            *[ls * num_examples for ls in ls_es], num_examples
        )
        pbar_q.put(1)
        # print(f"线程记录了第{epoch}世代的{n_data}批次的{which}数据")
        data = log_pc.recv()
        # n_data += 1
        # print(f"线程拿到了第{epoch}世代的{n_data}批次的{which}数据")
    pbar_q.put(f"世代{epoch}{which}记录完毕")