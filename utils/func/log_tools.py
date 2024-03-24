import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def write_log(path: str, **kwargs):
    """
    编写运行日志。
    :param path: 日志保存路径
    :param kwargs: 日志保存信息，类型为词典，key为列名，value为单元格内容
    :return: None
    """
    assert path.endswith('.csv'), f'日志文件格式为.csv，但指定的文件名为{path}'
    file_data = None
    wait = 0
    while file_data is None:
        try:
            file_data = pd.read_csv(path, encoding='utf-8')
        except FileNotFoundError:
            file_data = pd.DataFrame([])
        except PermissionError:
            for i in range(10):
                print(f'\r文件{path}被占用，已等待{wait}秒。请立即关闭此文件，在整10秒处会再次尝试获取文件读取权限。',
                      end='', flush=True)
                wait += 1
                time.sleep(1)
    item_data = pd.DataFrame([kwargs])
    if len(file_data) == 0:
        file_data = pd.DataFrame(item_data)
    else:
        file_data = pd.concat([file_data, item_data], axis=0, sort=True)
    wait = 0
    while True:
        try:
            file_data.to_csv(path, index=False, encoding='utf-8-sig')
            return
        except PermissionError:
            for i in range(10):
                print(f'\r 文件{path}被占用，已等待{wait}秒。请立即关闭此文件，在整10秒处会再次尝试获取文件读取权限。',
                      end='', flush=True)
                wait += 1
                time.sleep(1)


def plot_history(history,
                 mute=False, title=None,
                 savefig_as=None, accumulative=False,
                 max_nrows=3, figsize=(7, 7.5)):
    """绘制训练历史变化趋势图
    :param history: 训练历史数据
    :param mute: 绘制完毕后是否立即展示成果图
    :param title: 绘制图标题
    :param savefig_as: 保存图片路径
    :param accumulative: 是否将所有趋势图叠加在一起
    :param figsize: 图片大小
    :param max_nrows: 最大子图行数
    :return: None
    """
    # 获取子图标题以计算趋势图行列数
    subplots_titles = list(set([
        label.replace('train_', "").replace('valid_', '').upper()
        for label, _ in history
    ]))
    n_cols = int(np.ceil(len(subplots_titles) / max_nrows))
    fig, axes = plt.subplots(
        min(max_nrows, len(subplots_titles)), n_cols,
        sharex='col', figsize=(7, 7.5)
    )
    fig.set_figheight(figsize[1])
    fig.set_figwidth(figsize[0])
    # 设置子图横轴标签
    if n_cols > 1:
        for axi in axes.T:
            axi[-1].set_xlabel('epochs')
    else:
        axes[-1].set_xlabel('epochs')
    axes = axes.flatten()
    # 设置子图纵轴标签
    for subplots_title, axi in zip(subplots_titles, axes):
        axi.set_ylabel(subplots_title)
    # 绘制日志内容
    for label, log in history:
        l_type = label.replace('train_', "").replace('valid_', '').upper()
        index = subplots_titles.index(l_type)
        try:
            axes[index].plot(range(1, len(log) + 1), log, label=label)
        except ValueError:
            # 处理高维日志内容
            log = np.array(log)
            for i in range(len(log[0])):
                axes[index].plot(range(1, len(log) + 1), log[:, i], label=label + f'_{i}')
    if title:
        fig.suptitle(title)
    for ax in axes:
        ax.legend()
    if savefig_as:
        if not os.path.exists(os.path.split(savefig_as)[0]):
            os.makedirs(os.path.split(savefig_as)[0])
        plt.savefig(savefig_as)
        print('已保存历史趋势图')
    if not mute:
        plt.show()
    if not accumulative:
        plt.close(fig)
        plt.clf()


def get_logData(log_path, exp_no) -> dict:
    """
    通过实验编号获取实验数据。
    :param log_path: 实验文件路径
    :param exp_no: 实验编号
    :return: 实验数据字典`{数据名: 数据值}`
    """
    log = None
    wait = 0
    while log is None:
        try:
            log = pd.read_csv(log_path, encoding='utf-8')
            log = log.set_index('exp_no').to_dict('index')
            return log[exp_no]
        except KeyError:
            raise Exception(f'日志不存在{exp_no}项，请检查日志文件或重新选择查看的实验标号！')
        except FileNotFoundError:
            raise Exception(f'无法找到{log_path}文件，请检查日志文件路径是否输入正确！')
        except PermissionError:
            for i in range(10):
                print(f'\r文件{log_path}被占用，已等待{wait}秒。请立即关闭此文件，在整10秒处会再次尝试获取文件读取权限。',
                      end='', flush=True)
                wait += 1
                time.sleep(1)
