import time

import torch


@torch.no_grad()
def log_impl(
        pbar, preds, labels, ls_es, l_names, c_fns, c_names,
        metric_acc, durations, duration_acc
):
    logged_stamp = time.perf_counter()
    n_samples = len(preds)
    # 计算指标并记录
    metrics = [cfn(preds, labels) for cfn in c_fns]
    metric_acc.add(*metrics, *[ls * n_samples for ls in ls_es], n_samples)
    pbar.set_postfix(
        **{k: v.item() / n_samples for k, v in zip(l_names, ls_es)},
        **{k: v.item() / n_samples for k, v in zip(c_names, metrics)}
    )
    pbar.update(1)
    # 记录指标计算时间
    durations.append(time.perf_counter() - logged_stamp)
    duration_acc.add(*[d for d in durations], n_samples)


@torch.no_grad()
def log_multiprocessing_impl(
        epoch, which, log_q,
        c_fns, c_names, l_names, metric_acc,
        duration_acc, pbar_q
):
    """管理训练过程单个世代
    :param metric_acc: 主进程提供的数据累加器，只用于累加训练指标、损失值和样本数
    """
    n_data = 0
    data = log_q.get()
    # print(f"线程拿到了第{epoch}世代的{n_data}批次的{which}数据")
    logged_stamp = time.perf_counter()
    while data is not None:
        """训练数据记录"""
        preds, labels, ls_es, durations = data
        n_samples = len(preds)
        # 记录和计算指标，记录损失
        # 计算指标并记录
        metrics = [cfn(preds, labels) for cfn in c_fns]
        metric_acc.add(*metrics, *[ls * n_samples for ls in ls_es], n_samples)
        durations.append(time.perf_counter() - logged_stamp)
        duration_acc.add(*[d for d in durations], n_samples)
        logged_stamp = time.perf_counter()
        pbar_q.put({
            **{k: v.item() / n_samples for k, v in zip(l_names, ls_es)},
            **{k: v.item() / n_samples for k, v in zip(c_names, metrics)}
        })
        pbar_q.put(1)
        # print(f"线程记录了第{epoch}世代的{n_data}批次的{which}数据")
        data = log_q.get()
        n_data += 1
        # print(f"线程拿到了第{epoch}世代的{n_data}批次的{which}数据")
    # print(f"世代{epoch}的{which}记录线程退出")
    pbar_q.put(f"世代{epoch}{which}记录完毕")

def log_summarize(metric_acc, duration_acc, c_names, l_names, duration_names):
    if metric_acc[-1] > 0:
        metric_log = {
            name: metric_acc[i] / metric_acc[-1]
            for i, name in enumerate(c_names + l_names)
        }
        duration_log = {
            name: duration_acc[i] / duration_acc[-1]
            for i, name in enumerate(duration_names)
        }
    else:
        metric_log, duration_log = {}, {}
    return metric_log, duration_log