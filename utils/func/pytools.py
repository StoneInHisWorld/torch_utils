import math
import os.path
import warnings
from typing import Tuple, Iterable, Callable, Sized

from tqdm import tqdm

from utils.thread import Thread


def permutation(res: list, *args):
    """
    生成超参数列表。
    :param res: 结果列表。每个输出的列表都以`res`为前缀。
    :param args: 超参数列表。每个超参数输入均为列表，列表中的值为该超参数可能取值
    :return: 超参数取值列表生成器。
    """
    if len(args) == 0:
        yield res
    elif len(args) == 1:
        for arg in args[0]:
            yield res + [arg]
    else:
        for arg in args[0]:
            for p in permutation(res + [arg], *args[1:]):  # 不能直接yield本函数，否则不会执行
                yield p


def check_path(path: str, way_to_mkf=None):
    """
    检查指定路径。
    如果目录不存在，则会创建目录；
    如果文件不存在，则指定文件初始化方式后才会自动初始化文件
    :param path: 需要检查的目录
    :param way_to_mkf: 初始化文件的方法
    :return: None
    """
    if not os.path.exists(path):
        path, file = os.path.split(path)
        if file != "":
            # 如果是文件
            if way_to_mkf is not None:
                # 如果指定了文件初始化方式，则自动初始化文件
                if path == "" or os.path.exists(path):
                    way_to_mkf(file)
                else:
                    os.makedirs(path)
                    way_to_mkf(os.path.join(path, file))
            else:
                raise FileNotFoundError(f'没有在{path}下找到{file}文件！')
        else:
            # 如果目录不存在，则新建目录
            os.makedirs(path)


def check_para(name, value, val_range) -> bool:
    if value in val_range:
        return True
    else:
        warnings.warn(f'参数{name}需要取值限于{val_range}！')
        return False


def multi_process(n_workers: int = 8, mute: bool = True, desc: str = '',
                  *tasks: Tuple[Callable, Tuple, dict]) -> Iterable:
    results = []
    processors = []
    if mute:
        for task in tasks:
            func, args, kwargs = task
            t = Thread(func, *args, **kwargs)
            if len(processors) < n_workers:
                # 如果还有处理机，则运行当前线程
                t.start()
            else:
                # pop出队头处理机
                processor = processors.pop(0)
                # 等待线程运行完毕
                if processor.is_alive():
                    processor.join()
                # 获取运行结果
                results.append(processor.get_result())
                t.start()
            processors.append(t)
        for processor in processors:
            # 收集所有线程的运行结果
            if processor.is_alive():
                processor.join()
            results.append(processor.get_result())
    else:
        # 带有进度条的版本
        with tqdm(total=len(tasks), position=0, desc=desc, mininterval=1, unit='任务') as pbar:
            for task in tasks:
                func, args, kwargs = task
                t = Thread(func, *args, **kwargs)
                if len(processors) < n_workers:
                    t.start()
                else:
                    processor = processors.pop(0)
                    if processor.is_alive():
                        processor.join()
                    results.append(processor.get_result())
                    pbar.update(1)
                    t.start()
                processors.append(t)
            for processor in processors:
                if processor.is_alive():
                    processor.join()
                results.append(processor.get_result())
                pbar.update(1)
    return results


def iterable_multi_process(data: Iterable and Sized, task: Callable,
                           mute: bool = True, n_workers: int = 8, desc: str = '',
                           *args, **kwargs):
    data_l = len(data)
    batch_l = math.ceil(data_l / n_workers)
    data = [data[i: min(i + batch_l, data_l)] for i in range(0, data_l, batch_l)]
    ret = []
    for res in multi_process(
            n_workers, mute, desc, *[
                (task, (d, *args), kwargs)
                for d in data
            ]
    ):
        ret += res
    return ret
