import bisect
import threading
import time

from utils import ptools, History, Thread
from utils.accumulator import Accumulator
from . import _prepare_train, _prepare_valid, tduration_names, vduration_names
from .__hook_impl import hook
from .__log_impl import log_multiprocessing_impl, log_summarize
from .. import net_predict_state


debug = False


def __save_net(saver, net_q, save_msg_q):
    """使用bisect维护有序列表，无日志输出"""
    # 有序列表：元素为(epoch, net)，始终按epoch升序排列
    epoch_net_lst = []

    def process_save_msg(target_epoch: int, need_save: bool):
        # 1. 尝试在列表中找到目标epoch的网络
        # 用bisect快速查找目标epoch的位置
        idx = bisect.bisect_left(epoch_net_lst, (target_epoch,))
        # 2. 检查下标是否越界
        if idx < len(epoch_net_lst):
            # 2.1 没越界说明目标epoch的网络在列表中，进行保存操作
            epoch, net = epoch_net_lst.pop(idx)
        else:
            # 2.2.1 越界说明网络尚未发送，则等待队列数据
            try:
                epoch, net = net_q.get()
                if debug: print(f"收到了世代{epoch}的网络参数")
                while epoch != target_epoch:
                    # 2.2.2 世代数不正确则放进队列中
                    epoch_net_lst.append((epoch, net))
                    epoch, net = net_q.get()
                    if debug: print(f"收到了世代{epoch}的网络参数")
            except TypeError as e:
                if "cannot unpack non-iterable NoneType object" in str(e):
                    raise RuntimeError(f"出现错误，收到指标比较结果后没有找到对应世代{target_epoch}的网络！")
                else:
                    raise e
        # 2.3 世代数正确则考虑保存
        if need_save:
            if debug: print(f"保存了世代{target_epoch}的网络")
            saver.save_nocompare(net, epoch)


    # 主循环
    save_msg = save_msg_q.get()
    while save_msg is not None:
        if debug: print(f"收到了世代{save_msg[0]}的指标比较结果")
        process_save_msg(*save_msg)
        save_msg = save_msg_q.get()


@_prepare_train
@hook()
def train_and_valid(trainer,
                    tdata_q, vdata_q, pbar_q, epoch_q,
                    ctx, tdata_q_len, vdata_q_len,
                    result_conn):
    net = trainer.module
    # 传递训练数据的队列
    tlog_q = ctx.Queue(tdata_q_len)  # 训练数据
    vlog_q = ctx.Queue(vdata_q_len)  # 验证数据
    lrlog_q = ctx.Queue()  # 学习率
    log_epoch_q = ctx.Queue()  # 日志世代更新队列
    vepoch_q = ctx.Queue()  # 验证世代更新队列
    net_q = ctx.Queue()
    save_msg_q = ctx.Queue()
    # 设置网络是否进行训练事件
    training = threading.Event()
    training.set()

    # 创建记录进程，处理评价指标的计算以及记录，损失值、学习率的记录的事项
    log_subp = ctx.Process(
        target=__train_and_valid_log,
        args=(
            trainer.criterion_a, trainer.net_saver,
            net.train_ls_names, net.test_ls_names, net.lr_names,
            lrlog_q, tlog_q, vlog_q, log_epoch_q, pbar_q, save_msg_q,
            result_conn
        )
    )
    log_subp.start()
    # 挂载一个线程保存每一世代训练的网络，并接收log传来的讯息判断是否要保存某一世代的网络
    save_thread = Thread(__save_net, trainer.net_saver, net_q, save_msg_q)
    save_thread.start()
    # 挂在一个线程随时准备验证网络
    valid_thread = Thread(__valid, trainer, training, vdata_q, vlog_q, pbar_q, vepoch_q)
    valid_thread.start()

    logged_stamp = time.perf_counter()
    epoch = epoch_q.get()
    # 通过队列获取世代更新消息
    while epoch is not None:
        batch = tdata_q.get()
        pbar_q.put(f"世代{epoch}训练开始")
        log_epoch_q.put(epoch)
        n_batch = 0
        if debug: print(f"拿到世代{epoch}的{n_batch}批次的训练数据")
        # 等待训练允许
        training.wait()
        while batch is not None:
            """"训练实现"""
            # 拿到批量数据
            X, y = batch
            if debug: print(f"世代{epoch}的{n_batch}批次开始前反向传播")
            data_fetched_stamp = time.perf_counter()
            # 前反向传播
            pred_s, ls_es = net.forward_backward(X, y)
            if debug: print(f"世代{epoch}的{n_batch}批次前反向传播已经完成")
            fb_ward_stamp = time.perf_counter()
            # 数据传递给记录进程，进行评价指标计算、历史记录更新（损失值、评价指标、学习率）
            durations = [data_fetched_stamp - logged_stamp, fb_ward_stamp - data_fetched_stamp]
            log_data = n_batch, pred_s.detach().clone(), y.detach().clone(), [l.detach().clone() for l in ls_es], durations
            tlog_q.put(log_data)
            if debug: print(f"世代{epoch}的{n_batch}批次训练结果已经发送")
            logged_stamp = time.perf_counter()
            # 获取下一批量数据
            batch = tdata_q.get()
            n_batch += 1
            if batch is not None and debug: print(f"拿到世代{epoch}的{n_batch}批次的训练数据")
        lrlog_q.put(net.get_lr_groups())
        # 更新优化器学习率
        net.update_lr()
        tlog_q.put(None)
        pbar_q.put(f"世代{epoch}训练完毕")
        # 验证并通知网络保存线程
        vepoch_q.put(epoch)
        net_q.put((epoch, net))
        if debug: print(f"世代{epoch}的网络参数已发送")
        # 等待下一世代开始信号
        epoch = epoch_q.get()


    # 记录最后一次学习率，并通知学习率已经记录完毕
    log_epoch_q.put(None)
    lrlog_q.put(None)
    vepoch_q.put(None)
    result_conn.send(net)
    # 等待所有进程、线程执行完毕
    log_subp.join()
    save_thread.join()
    valid_thread.join()
    if debug: print("记录进程结束")

# @_prepare_train
# @hook()
# def train_and_valid(trainer,
#                     tdata_q, vdata_q, pbar_q, epoch_q,
#                     ctx, tdata_q_len, vdata_q_len,
#                     result_conn):
#     net = trainer.module
#     # 传递训练数据的队列
#     tlog_q = ctx.Queue(tdata_q_len)  # 训练数据
#     vlog_q = ctx.Queue(vdata_q_len)  # 验证数据
#     lrlog_q = ctx.Queue()  # 学习率
#     log_epoch_q = ctx.Queue()  # 日志世代更新队列
#     vepoch_q = ctx.Queue()  # 验证世代更新队列
#     net_q = ctx.Queue()
#     save_msg_q = ctx.Queue()
#
#     # 创建记录进程，处理评价指标的计算以及记录，损失值、学习率的记录的事项
#     log_subp = ctx.Process(
#         target=__train_and_valid_log,
#         args=(
#             trainer.criterion_a, trainer.net_saver,
#             net.train_ls_names, net.test_ls_names, net.lr_names,
#             lrlog_q, tlog_q, vlog_q, log_epoch_q, pbar_q, save_msg_q,
#             result_conn
#         )
#     )
#     log_subp.start()
#     # 挂载一个线程保存每一世代训练的网络，并接收log传来的讯息判断是否要保存某一世代的网络
#     save_thread = Thread(__save_net, trainer.net_saver, net_q, save_msg_q)
#     save_thread.start()
#
#     logged_stamp = time.perf_counter()
#     epoch = epoch_q.get()
#     # 通过队列获取世代更新消息
#     while epoch is not None:
#         batch = tdata_q.get()
#         pbar_q.put(f"世代{epoch}训练开始")
#         log_epoch_q.put(epoch)
#         n_batch = 0
#         if debug:
#             print(f"拿到世代{epoch}的{n_batch}批次的训练数据")
#         while batch is not None:
#             """"训练实现"""
#             # 拿到批量数据
#             X, y = batch
#             if debug:
#                 print(f"世代{epoch}的{n_batch}批次开始前反向传播")
#             data_fetched_stamp = time.perf_counter()
#             # 前反向传播
#             pred_s, ls_es = net.forward_backward(X, y)
#             if debug:
#                 print(f"世代{epoch}的{n_batch}批次前反向传播已经完成")
#             fb_ward_stamp = time.perf_counter()
#             # 数据传递给记录进程，进行评价指标计算、历史记录更新（损失值、评价指标、学习率）
#             durations = [data_fetched_stamp - logged_stamp, fb_ward_stamp - data_fetched_stamp]
#             log_data = n_batch, pred_s.detach().clone(), y.detach().clone(), [l.detach().clone() for l in ls_es], durations
#             tlog_q.put(log_data)
#             if debug:
#                 print(f"世代{epoch}的{n_batch}批次训练结果已经发送")
#             logged_stamp = time.perf_counter()
#             # 获取下一批量数据
#             batch = tdata_q.get()
#             n_batch += 1
#             if batch is not None and debug:
#                 print(f"拿到世代{epoch}的{n_batch}批次的训练数据")
#         lrlog_q.put(net.get_lr_groups())
#         # 更新优化器学习率
#         net.update_lr()
#         tlog_q.put(None)
#         pbar_q.put(f"世代{epoch}训练完毕")
#         # 验证
#         __valid(trainer, vdata_q, vlog_q, pbar_q, epoch)
#         net_q.put((epoch, net))
#         if debug: print(f"世代{epoch}的网络参数已发送")
#         epoch = epoch_q.get()
#
#
#     # 记录最后一次学习率，并通知学习率已经记录完毕
#     log_epoch_q.put(None)
#     lrlog_q.put(None)
#     result_conn.send(net)
#     # 等待所有进程、线程执行完毕
#     log_subp.join()
#     save_thread.join()
#     if debug: print("记录进程结束")


@_prepare_valid
def __valid(trainer, training, vdata_q, vlog_q, pbar_q, vepoch_q):
    """验证函数实现
    每次取出验证数据供给器中的下一批次数据进行前向传播，之后计算评价指标和损失，生成验证日志。

    :param valid_iter: 验证数据供给器
    :return: 验证记录
    """
    # 提取出验证网络
    net = trainer.module
    # 获取世代信号作为本世代的验证开始标志
    epoch = vepoch_q.get()
    while epoch is not None:
        # 获取网络反向传播锁，避免网络参数被更新
        training.clear()
        pre_state = net.state
        net.state = net_predict_state
        if debug: print("训练信号设置为False")
        pbar_q.put(f"世代{epoch}开始验证")
        n_batch = 0
        # 计时：数据获取
        logged_stamp = time.perf_counter()
        batch = vdata_q.get()
        while batch is not None:
            X, y = batch  # epoch应该每次都相同
            if debug: print(f"拿到世代{epoch}的{n_batch}批次的验证数据")
            data_fetched_stamp = time.perf_counter()
            # 计时：前反向传播
            pred_s, ls_es = net.forward_backward(X, y)
            predicted_stamp = time.perf_counter()
            # 计时：数据传递给记录进程，进行评价指标计算、历史记录更新（损失值、评价指标、学习率）
            durations = [
                data_fetched_stamp - logged_stamp,
                predicted_stamp - data_fetched_stamp
            ]
            log_data = (n_batch, pred_s.detach().clone(), y.detach().clone(),
                        [l.detach().clone() for l in ls_es], durations)
            vlog_q.put(log_data)
            if debug: print(f"世代{epoch}的{n_batch}批次的验证前向传播数据已经发送")
            logged_stamp = time.perf_counter()
            # 计时：获取下一批数据
            batch = vdata_q.get()
            n_batch += 1
        # 验证完毕，通知训练线程可以开始训练了，并将网络状态复原
        training.set()
        net.state = pre_state
        if debug: print("训练信号设置为True")
        # 通知记录进程，本世代的验证前向传播已经计算完毕
        vlog_q.put(None)
        pbar_q.put(f"世代{epoch}验证完毕")
        # 等待下一次验证
        epoch = vepoch_q.get()


# @_prepare_valid
# def __valid(trainer, vdata_q, vlog_q, pbar_q, epoch):
#     """验证函数实现
#     每次取出验证数据供给器中的下一批次数据进行前向传播，之后计算评价指标和损失，生成验证日志。
#
#     :param valid_iter: 验证数据供给器
#     :return: 验证记录
#     """
#     # 提取出验证所需参数
#     net = trainer.module
#     # 获取数据
#     logged_stamp = time.perf_counter()
#     batch = vdata_q.get()
#     n_batch = 0
#     while batch is not None:
#         X, y = batch  # epoch应该每次都相同
#         data_fetched_stamp = time.perf_counter()
#         pred_s, ls_es = net.forward_backward(X, y)
#         # 数据传递给记录进程，进行评价指标计算、历史记录更新（损失值、评价指标、学习率）
#         predicted_stamp = time.perf_counter()
#         durations = [
#             data_fetched_stamp - logged_stamp,
#             predicted_stamp - data_fetched_stamp
#         ]
#         log_data = (n_batch, pred_s.detach().clone(), y.detach().clone(),
#                     [l.detach().clone() for l in ls_es], durations)
#         vlog_q.put(log_data)
#         if debug:
#             print(f"世代{epoch}的{n_batch}批次的验证数据已经发送")
#         logged_stamp = time.perf_counter()
#         # 获取下一批数据
#         batch = vdata_q.get()
#         n_batch += 1
#         if debug:
#             print(f"拿到世代{epoch}的{n_batch}批次的验证数据")
#     pbar_q.put(f"世代{epoch}验证完毕")


def __train_and_valid_log(
        criteria_fns, net_saver,
        trls_names, tels_names, lr_names,
        lrlog_q, tlog_q, vlog_q, epoch_q, pbar_q, save_msg_q,
        result_conn
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
    # 接收世代更新消息
    epoch = epoch_q.get()
    if debug:
        print(f"开始记录世代{epoch}")
    while epoch is not None:
        # 创建训练数据记录线程
        tmetric_acc.reset()
        tlog_thread = Thread(
            log_multiprocessing_impl,
            epoch, "训练", criteria_fns, tc_names, tl_names,
            tmetric_acc, tduration_acc, tlog_q, pbar_q
        )
        tlog_thread.start()
        # 创建验证数据记录线程
        vmetric_acc.reset()
        vlog_thread = Thread(
            log_multiprocessing_impl,
            epoch, "验证", criteria_fns, vc_names, vl_names,
            vmetric_acc, vduration_acc, vlog_q, pbar_q
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
        tmetric_log, tduration_log = log_summarize(
            tmetric_acc, tduration_acc,
            tc_names, tl_names, tduration_names
        )
        vmetric_log, vduration_log = log_summarize(
            vmetric_acc, vduration_acc,
            vc_names, vl_names, vduration_names
        )
        # metric_history.add(
        #     list(filter(lambda k: "_lrs" not in k, history_keys)), [
        #         *[tmetric_acc[i] / tmetric_acc[-1] for i in range(len(tmetric_acc) - 1)],
        #         *[vmetric_acc[i] / vmetric_acc[-1] for i in range(len(vmetric_acc) - 1)]
        #     ]
        # )
        # duration_history.add(
        #     tduration_names + vduration_names,
        #     [tduration_acc[i] / tduration_acc[-1] for i in range(len(tduration_acc) - 1)] +
        #     [vduration_acc[i] / vduration_acc[-1] for i in range(len(vduration_acc) - 1)]
        # )
        metric_history.add(
            list(tmetric_log.keys()) + list(vmetric_log.keys()),
            list(tmetric_log.values()) + list(vmetric_log.values())
        )
        duration_history.add(
            list(tduration_log.keys()) + list(vduration_log.keys()),
            list(tduration_log.values()) + list(vduration_log.values())
        )
        save_msg_q.put((epoch, net_saver.compare_update_record(vmetric_log)))
        pbar_q.put(f"世代{epoch}记录完毕")
        epoch = epoch_q.get()
    # 通知各个进程、线程，训练和验证都已执行完毕
    pbar_q.put(None)
    save_msg_q.put(None)
    result_conn.send(metric_history)
    result_conn.send(duration_history)
    if debug:
        print("历史记录发送完毕")
