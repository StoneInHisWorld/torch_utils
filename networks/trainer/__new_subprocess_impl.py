import time

from torch.multiprocessing import Process

from utils import ptools, History, Thread
from utils.accumulator import Accumulator
from . import _prepare_train, _prepare_valid, tduration_names, vduration_names
from .__hook_impl import hook
from .__log_impl import log_multiprocessing_impl


debug = False


@_prepare_train
@hook()
def train_valid_impl(trainer, 
                     tdata_q, vdata_q, pbar_q, epoch_q, 
                     ctx, tdata_q_len, vdata_q_len,
                     result_conn):
    net = trainer.module
    # 传递训练数据的队列
    tlog_q = ctx.Queue(tdata_q_len)  # 训练数据
    vlog_q = ctx.Queue(vdata_q_len)  # 验证数据
    lrlog_q = ctx.Queue()  # 学习率
    log_epoch_q = ctx.Queue()  # 世代更新队列
    # 创建记录进程，处理评价指标的计算以及记录，损失值、学习率的记录的事项
    log_subp = Process(
        target=__train_and_valid_logging,
        args=(trainer.criterion_a, net.train_ls_names, net.test_ls_names, net.lr_names,
              lrlog_q, tlog_q, vlog_q, log_epoch_q, pbar_q, result_conn)
    )
    log_subp.start()
    # 通过队列获取世代更新消息。这里不能写成while q.get(): 因为这里的世代数会等于0
    logged_stamp = time.perf_counter()
    epoch = epoch_q.get()
    while epoch is not None:
        batch = tdata_q.get()
        pbar_q.put(f"世代{epoch}训练开始")
        log_epoch_q.put(epoch)
        n_batch = 0
        if debug:
            print(f"拿到世代{epoch}的{n_batch}批次的训练数据")
        while batch is not None:
            """"训练实现"""
            # 拿到批量数据
            X, y = batch
            if debug:
                print(f"世代{epoch}的{n_batch}批次开始前反向传播")
            data_fetched_stamp = time.perf_counter()
            # 前反向传播
            pred_s, ls_es = net.forward_backward(X, y)
            if debug:
                print(f"世代{epoch}的{n_batch}批次前反向传播已经完成")
            fb_ward_stamp = time.perf_counter()
            # 数据传递给记录进程，进行评价指标计算、历史记录更新（损失值、评价指标、学习率）
            durations = [data_fetched_stamp - logged_stamp, fb_ward_stamp - data_fetched_stamp]
            log_data = pred_s.detach().clone(), y.detach().clone(), [l.detach().clone() for l in ls_es], durations
            tlog_q.put(log_data)
            if debug:
                print(f"世代{epoch}的{n_batch}批次训练结果已经发送")
            logged_stamp = time.perf_counter()
            # 获取下一批量数据
            batch = tdata_q.get()
            n_batch += 1
            if batch is not None and debug:
                print(f"拿到世代{epoch}的{n_batch}批次的训练数据")
        lrlog_q.put(net.get_lr_groups())
        # print(f"世代{epoch}的学习率组已经发送")
        # 更新优化器学习率
        net.update_lr()
        tlog_q.put(None)
        pbar_q.put(f"世代{epoch}训练完毕")
        # 验证
        __valid_impl(trainer, vdata_q, vlog_q, pbar_q, epoch)
        epoch = epoch_q.get()
    # 记录最后一次学习率，并通知学习率已经记录完毕
    log_epoch_q.put(None)
    lrlog_q.put(None)
    # # 使用None来通知记录进程“训练已经结束”
    # result_conn.send(net)
    # if debug:
    #     print("网络已发送")
    log_subp.join()
    if debug:
        print("记录进程结束")


@_prepare_valid
def __valid_impl(trainer, vdata_q, vlog_q, pbar_q, epoch):
    """验证函数实现
    每次取出验证数据供给器中的下一批次数据进行前向传播，之后计算评价指标和损失，生成验证日志。

    :param valid_iter: 验证数据供给器
    :return: 验证记录
    """
    # 提取出验证所需参数
    net = trainer.module
    # 获取数据
    logged_stamp = time.perf_counter()
    batch = vdata_q.get()
    n_batch = 0
    # pbar_q.put(f"世代{epoch}验证开始")
    while batch is not None:
        X, y = batch  # epoch应该每次都相同
        data_fetched_stamp = time.perf_counter()
        pred_s, ls_es = net.forward_backward(X, y, False)
        # 数据传递给记录进程，进行评价指标计算、历史记录更新（损失值、评价指标、学习率）
        predicted_stamp = time.perf_counter()
        durations = [
            data_fetched_stamp - logged_stamp,
            predicted_stamp - data_fetched_stamp
        ]
        log_data = (pred_s.detach().clone(), y.detach().clone(),
                    [l.detach().clone() for l in ls_es], durations)
        vlog_q.put(log_data)
        if debug:
            print(f"世代{epoch}的{n_batch}批次的验证数据已经发送")
        logged_stamp = time.perf_counter()
        # 获取下一批数据
        batch = vdata_q.get()
        n_batch += 1
        if debug:
            print(f"拿到世代{epoch}的{n_batch}批次的验证数据")
    # vlog_q.put(None)
    # pbar_q.put(f"世代{epoch}验证完毕")


def __train_and_valid_logging(
        criteria_fns, trls_names, tels_names, lr_names,
        lrlog_q, tlog_q, vlog_q, epoch_q, pbar_q, result_conn
):
    """管理整个训练过程的历史记录"""
    # 指标项名称
    tc_names = [f'train_{ptools.get_computer_name(cfn)}' for cfn in criteria_fns]
    tl_names = [f'train_{ln}' for ln in trls_names]
    vc_names = [f'valid_{ptools.get_computer_name(cfn)}' for cfn in criteria_fns]
    vl_names = [f'valid_{ln}' for ln in tels_names]
    lr_names = [f'{lr}_lrs' for lr in lr_names]
    # 创建历史记录
    history_keys = tc_names + tl_names + vc_names + vl_names + lr_names
    metric_history = History(*history_keys)
    duration_history = History(*(tduration_names + vduration_names))

    # 创建线程进行学习率的记录
    def add_lr_to_history():
        data = lrlog_q.get()
        while data is not None:
            lr_names, lrs = data
            metric_history.add([f"{ln}_lrs" for ln in lr_names], lrs)
            data = lrlog_q.get()

    lrlog_thread = Thread(add_lr_to_history)
    lrlog_thread.start()
    # 每次世代开始就创建训练数据记录线程和验证数据记录线程，得到其结果后记录到历史记录对象中
    tmetric_acc = Accumulator(len(tl_names + tc_names) + 1)
    vmetric_acc = Accumulator(len(vl_names + vc_names) + 1)
    tduration_acc = Accumulator(len(tduration_names) + 1)
    vduration_acc = Accumulator(len(vduration_names) + 1)
    # 接收世代更新消息。这里不能写成while q.get(): 因为这里的世代数会等于0
    epoch = epoch_q.get()
    if debug:
        print(f"开始记录世代{epoch}")
    while epoch is not None:
        # 创建训练数据记录线程
        tmetric_acc.reset()
        tlog_thread = Thread(
            log_multiprocessing_impl,
            epoch, "训练", tlog_q, criteria_fns, tc_names, tl_names,
            tmetric_acc, tduration_acc, pbar_q
        )
        tlog_thread.start()
        # 创建验证数据记录线程
        vmetric_acc.reset()
        vlog_thread = Thread(
            log_multiprocessing_impl,
            epoch, "验证", vlog_q, criteria_fns, vc_names, vl_names,
            vmetric_acc, vduration_acc, pbar_q
        )
        vlog_thread.start()
        # 等待数据记录完毕
        tlog_thread.join()
        if debug:
            print(f"世代{epoch}训练数据计算完毕")
        vlog_thread.join()
        if debug:
            print(f"世代{epoch}验证数据计算完毕")
        # 统计指标数据并加入到历史记录中
        metric_history.add(
            list(filter(lambda k: "_lrs" not in k, history_keys)), [
                *[tmetric_acc[i] / tmetric_acc[-1] for i in range(len(tmetric_acc) - 1)],
                *[vmetric_acc[i] / vmetric_acc[-1] for i in range(len(vmetric_acc) - 1)]
            ]
        )
        duration_history.add(
            tduration_names + vduration_names,
            [tduration_acc[i] / tduration_acc[-1] for i in range(len(tduration_acc) - 1)] +
            [vduration_acc[i] / vduration_acc[-1] for i in range(len(vduration_acc) - 1)]
        )
        pbar_q.put(f"世代{epoch}记录完毕")
        epoch = epoch_q.get()
    pbar_q.put(None)
    result_conn.send(metric_history)
    result_conn.send(duration_history)
    result_conn.close()
    if debug:
        print("历史记录发送完毕")
