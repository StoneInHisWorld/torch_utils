import os
import time

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
                      end='',  flush=True)
                wait += 1
                time.sleep(1)


def plot_history(history, mute=False, title=None,
                 ls_ylabel=None, acc_ylabel=None, lr_ylabel=None,
                 savefig_as=None, accumulative=False):
    """
    绘制训练历史变化趋势图
    :param acc_ylabel: 准确率趋势图的纵轴标签
    :param ls_ylabel: 损失值趋势图的纵轴标签
    :param history: 训练历史数据
    :param mute: 绘制完毕后是否立即展示成果图
    :param title: 绘制图标题
    :param savefig_as: 保存图片路径
    :param accumulative: 是否将所有趋势图叠加在一起
    :return: None
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col', figsize=(7, 7.5))
    ax1.set_title('LOSS')
    ax2.set_title('ACCURACY')
    ax3.set_title('LEARNING_RATE')
    ax3.set_xlabel('epochs')
    for label, log in history:
        if label.find('_l') != -1:
            # 绘制损失值history
            ax1.plot(range(1, len(log) + 1), log, label=label)
        elif label.find('_acc') != -1:
            # 绘制准确率history
            ax2.plot(range(1, len(log) + 1), log, label=label)
        elif label.find('lr') != -1:
            # 绘制学习率history
            for i in range(len(log[0])):
                lr_of_cur_group = [epoch_lrs[i] for epoch_lrs in log]
                ax3.plot(range(1, len(log) + 1), lr_of_cur_group, label=f'lr_{i}')
    if ls_ylabel:
        ax1.set_ylabel(ls_ylabel)
    if acc_ylabel:
        ax2.set_ylabel(acc_ylabel)
    if lr_ylabel:
        ax3.set_ylabel(lr_ylabel)
    if title:
        fig.suptitle(title)
    ax1.legend()
    ax2.legend()
    ax3.legend()
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
    # TODO：Untested!
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
    # try:
    #     log = pd.read_csv(log_path)
    #     log = log.set_index('exp_no').to_dict('index')
    #     return log[exp_no]
    # except KeyError:
    #     raise Exception(f'日志不存在{exp_no}项，请检查日志文件或重新选择查看的实验标号！')
    # except FileNotFoundError:
    #     raise Exception(f'无法找到{log_path}文件，请检查日志文件路径是否输入正确！')
