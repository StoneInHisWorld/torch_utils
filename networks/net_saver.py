import os
import pickle
from io import BytesIO

import torch

from utils import ptools


save_net_range = ['no', 'entire', 'state']
debug = False


class NetSaver:

    def __init__(self, exp_no, save_root, save_format):
        self.save_root = save_root
        self.exp_no = exp_no
        assert save_format in save_net_range, (f'请检查setting.json中参数save_net设置是否正确，'
                                               f'可设置取值为：{save_net_range}!')
        self.save_format = save_format

    def save_nocompare(self, net, n_epoch = ""):
        # 根据设置的网络保存要求进行网络的保存
        assert isinstance(net, torch.nn.Module), f"输入的网络需要为torch.nn.Module对象！"
        if self.save_format == 'entire':
            obj_to_be_saved, posfix = net, ".ptm"
        elif self.save_format == 'state':
            obj_to_be_saved, posfix = net.state_dict(), ".ptsd"
        else:
            raise ValueError(f"收到了不正确的网络保存格式{self.save_format}！")
        save_path = os.path.join(self.save_root, f'{self.exp_no}_epoch{n_epoch}{posfix}')
        # 删除上次保存的网络文件，用更好的网络文件代替
        if hasattr(self, "last_save_path"):
            os.remove(self.last_save_path)
            if debug: print(f"删除网络{self.last_save_path}")
        self.last_save_path = save_path
        # 保存更好的网络文件
        torch.save(obj_to_be_saved, save_path)
        if debug: print(f"保存网络{save_path}")

    def compare_update_record(self, record):
        # 对结果进行比较，如果当前结果更好则进行接下来的保存，否则退出
        if not hasattr(self, "best_record"):
            pass
        elif not self.compare(self.best_record, record):
            return False
        self.best_record = record
        return True

    def save(self, net, record, n_epoch = ""):
        """保存实验对象持有网络
        根据动态运行参数进行相应的网络保存动作，具有三种保存模式，保存模式由动态运行参数save_net指定：
        entire：指持久化整个网络对象
        state：指持久化网络对象参数
        no：指不进行持久化

        :return: None
        """
        self.compare_update_record(record)
        self.save_nocompare(net, n_epoch)
        # assert isinstance(net, torch.nn.Module), f"输入的网络需要为torch.nn.Module对象！"
        # # 对结果进行比较，如果当前结果更好则进行接下来的保存，否则退出
        # if not hasattr(self, "best_record"):
        #     pass
        # elif not self.compare(self.best_record, record):
        #     return
        # self.best_record = record
        # # 根据设置的网络保存要求进行网络的保存
        # if self.save_format == 'entire':
        #     obj_to_be_saved, posfix = net, ".ptm"
        # elif self.save_format == 'state':
        #     obj_to_be_saved, posfix = net.state_dict(), ".ptsd"
        # else:
        #     raise ValueError(f"收到了不正确的网络保存格式{self.save_format}！")
        # torch.save(obj_to_be_saved, os.path.join(self.save_root,
        #                                          f'{self.exp_no}_epoch{n_epoch}{posfix}'))
        # if debug:
        #     print(f"保存网络{os.path.join(self.save_root, f'{self.exp_no}_epoch{n_epoch}{posfix}')}")

    @property
    def compare(self):
        assert hasattr(self, "_compare_fn"), ("请通过compare属性给NetSaver赋值自定义比较函数，用于比较两次训练结果以决定是否保存网络！"
                                               "比较函数签名需要为：def compare_fn(best_record, record) -> bool，"
                                               "其中两个record参数均为字典。")
        return self._compare_fn

    @compare.setter
    def compare(self, fn):
        try:
            # 创建字节流缓冲区，用于临时存储序列化数据
            buffer = BytesIO()
            # 尝试序列化对象（使用最高协议以兼容更多类型）
            pickle.dump(fn, buffer, protocol=pickle.HIGHEST_PROTOCOL)
            # 可选：验证反序列化是否正常（确保序列化后的对象可恢复）
            buffer.seek(0)
            pickle.load(buffer)
        except (pickle.PicklingError, AttributeError, TypeError, ImportError) as e:
            # 捕获常见的序列化失败异常
            raise ValueError(f"请设置可以被序列化的比较方法！")
        _, pos_or_kwargs, _, _ = ptools.get_signature(fn)
        assert len(
            pos_or_kwargs) == 2, f"比较方法接受的参数需要为两个字典对象！检测到输入的比较方法签名中，需要的位置参数数量为{len(pos_or_kwargs)}"
        self._compare_fn = fn
