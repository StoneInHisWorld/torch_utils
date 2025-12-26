import functools
from io import BytesIO
import json
import math
import os.path
<<<<<<< HEAD
=======
import pickle
>>>>>>> fe67262f0be35bf6395172a41ccd9efe30baa9c9
import re
import warnings

import jsonref
import pandas as pd
import torch
from jsonref import JsonRef

<<<<<<< HEAD
from config.init_cfg import init_log, init_predict_settings, init_train_settings, init_hps
=======
from config.init_cfg import init_predict_settings, init_train_settings, init_hps
>>>>>>> fe67262f0be35bf6395172a41ccd9efe30baa9c9
from .experiment import New2Experiment
from .func import pytools as ptools
from .func import log_tools as ltools


def resolve_jsonref(obj):
    """用于处理JsonRef对象"""
    if isinstance(obj, JsonRef):
        return resolve_jsonref(obj.__subject__)  # 解引用
    elif isinstance(obj, dict):
        return {k: resolve_jsonref(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_jsonref(item) for item in obj]
    else:
        return obj
    
def get_req_sha(req_sha):
    if isinstance(req_sha, float):
        if math.isnan(req_sha):
            return None
    if isinstance(req_sha, str):
        dims = re.findall(r'\d+', req_sha)
        req_sha = tuple(map(int, dims))
    return req_sha


save_net_range = ['no', 'entire', 'state']
plot_history_range = ['plot', 'save', 'no']


<<<<<<< HEAD
=======
def save_net_fn(
    net, before_log, after_log,
    compare, exp_no, save_path, save_format
):
    """保存实验对象持有网络
    根据动态运行参数进行相应的网络保存动作，具有三种保存模式，保存模式由动态运行参数save_net指定：
    entire：指持久化整个网络对象
    state：指持久化网络对象参数
    no：指不进行持久化

    :return: None
    """
    if compare(before_log, after_log):
        assert isinstance(net, torch.nn.Module), f"输入的网络需要为torch.nn.Module对象！"
        if save_format == 'entire':
            obj_to_be_saved, posfix = net, ".ptm"
        elif save_format == 'state':
            obj_to_be_saved, posfix = net.state_dict(), ".ptsd"
        else:
            raise ValueError(f"收到了不正确的网络保存格式{save_format}！")
        torch.save(obj_to_be_saved, os.path.join(save_path, f'{exp_no}{posfix}'))


>>>>>>> fe67262f0be35bf6395172a41ccd9efe30baa9c9
class New2ControlPanel:
    """控制台类负责读取、管理动态运行参数、超参数组合，以及实验对象的提供"""

    def __init__(self, datasource, net_type, is_train, cfg_root=os.path.join(".", "config")):
        """控制面板

        负责读取、管理动态运行参数、超参数组合，生成超参数配置文件路径、日志文件存储路径以及网络文件存储目录，提供实验Experiment对象。
        迭代每次提供一个实验对象，包含有单次训练的超参数组合以及动态运行参数组合，训练过程中不可改变，每次迭代后会更新运行配置参数。
        :param datasource: 训练数据来源
        :param net_type: 训练所用模型类
        :param cfg_root: 运行配置文件路径
        """
        net_name = net_type.__name__.lower()
        self.net_type = net_type
        # 生成运行动态配置
        self.__rcp = os.path.join(cfg_root, net_name, f'settings.json')  # 运行配置json文件路径
<<<<<<< HEAD
        # log_root = self.cfg_dict['log_root']
=======
>>>>>>> fe67262f0be35bf6395172a41ccd9efe30baa9c9
        if is_train:
            self.__train_init(cfg_root, net_name)
        else:
            self.__predict_init(net_name)
        # 读取运行配置
        self.__read_runtime_cfg()
<<<<<<< HEAD
        # if is_train:
        #     ptools.check_path(self.__rcp, init_train_settings)
        # else:
        #     ptools.check_path(self.__rcp, init_predict_settings)
        # self.__read_runtime_cfg()
        # 生成其他路径
        # self.__hcp = os.path.join(cfg_root, net_name, f'hyper_param_s.json')  # 网络训练超参数文件路径
        # self.log_root = self.cfg_dict['log_root']
        # if is_train:
        #     self.__mlp = os.path.join(self.log_root, net_name, 'metric_log.csv')  # 指标日志文件存储路径
        #     self.__plp = os.path.join(self.log_root, net_name, 'perf_log.csv')  # 性能日志文件存储路径
        #     self.__np = os.path.join(self.log_root, net_name, 'trained_net', "")  # 训练成果网络存储路径
        #     self.__pp = os.path.join(self.log_root, net_name, 'imgs', "")  # 历史趋势图存储路径
        #     # 路径检查
        #     ptools.check_path(self.__hcp, init_hps)
        #     ptools.check_path(self.__mlp, init_log)
        #     ptools.check_path(self.__plp, init_log)
        #     ptools.check_path(self.__np)
        #     ptools.check_path(self.__pp)
        # else:
        #     self.__mlp = os.path.join(self.log_root, net_name, 'metric_log.csv')  # 指标日志文件存储路径
        #     self.__plp = os.path.join(self.log_root, net_name, 'perf_log.csv')  # 性能日志文件存储路径
        #     self.__np = os.path.join(self.log_root, net_name, 'trained_net', "")  # 训练成果网络存储路径
        #     self.__pp = os.path.join(self.log_root, net_name, 'imgs', "")  # 历史趋势图存储路径
        #     # 路径检查
        #     ptools.check_path(self.__hcp, init_hps)
        #     ptools.check_path(self.__mlp, init_log)
        #     ptools.check_path(self.__plp, init_log)
        #     ptools.check_path(self.__np)
        #     ptools.check_path(self.__pp)
        # # # 读取实验编号
        # self.__read_expno()
=======
>>>>>>> fe67262f0be35bf6395172a41ccd9efe30baa9c9
        self.dao_ds = datasource
        self.is_train = is_train
        self.plot_kwargs = {}

    def __train_init(self, cfg_root, net_name):
                # 读取运行配置
        ptools.check_path(self.__rcp, init_train_settings)
        self.__read_runtime_cfg()
        # 生成其他路径
        self.__hcp = os.path.join(cfg_root, net_name, f'hyper_param_s.json')  # 网络训练超参数文件路径
        log_root = self.cfg_dict['log_root']
        self.__mlp = os.path.join(log_root, net_name, 'metric_log.csv')  # 指标日志文件存储路径
        self.__plp = os.path.join(log_root, net_name, 'perf_log.csv')  # 性能日志文件存储路径
        self.__np = os.path.join(log_root, net_name, 'trained_net', "")  # 训练成果网络存储路径
        self.__pp = os.path.join(log_root, net_name, 'imgs', "")  # 历史趋势图存储路径
        # 路径检查
        ptools.check_path(self.__hcp, init_hps)
        ptools.check_path(self.__np)
        ptools.check_path(self.__pp)
        # 设置随机种子
        torch.random.manual_seed(self['random_seed'])
        
    def __predict_init(self, net_name):
        """预测模式初始化"""
        # 读取运行配置
        ptools.check_path(self.__rcp, init_predict_settings)
        self.__read_runtime_cfg()
        log_root = self.cfg_dict['log_root']
        # 生成其他路径
        self.__mlp = os.path.join(log_root, net_name, 'metric_log.csv')  # 指标日志文件存储路径
        self.__np = os.path.join(log_root, net_name, 'trained_net', "")  # 训练成果网络存储路径
        self.__wrp = os.path.join(log_root, net_name, 'results', "")  # 预测结果存储路径 

    def __read_expno(self):
        """读取实验编号
        从日志中读取最后一组实验数据的实验编号，从而推算出即将进行的所有实验组对应的编号。
        本函数将会创建self.exp_no以及self.last_expno属性。
        :return: None
        """
        # 读取性能日志的最后一项，取出实验编号作为本次实验编号的参考
        try:
            log = pd.read_csv(self.__plp)
            exp_no = log.iloc[-1]['exp_no'] + 1
        except Exception as _:
            exp_no = 1
        assert exp_no > 0, f'训练序号需为正整数，但读取到的序号为{exp_no}'
        self.exp_no = int(exp_no)
        # 计算总共需要进行的实验组数
        with open(self.__hcp, 'r', encoding='utf-8') as cfg:
            hyper_params = json.load(cfg)
            n_exp = 1
            for v in hyper_params.values():
                n_exp *= len(v)
        # 最后一组实验的实验编号
        self.last_expno = self.exp_no + n_exp - 1
        
    @property
    def reading_queue(self):
        return self.__reading_queue
    
    @reading_queue.setter
    def reading_queue(self, queue):
        self.__reading_queue = queue
<<<<<<< HEAD
        
    # def set_reading_queue(self, *queue):
    #     """设置读取实验编号队列
    #     该方法用于设置读取实验编号的队列，通常用于预测模式下的实验编号读取。
    #     :param queue: 实验编号队列
    #     :return: None
    #     """
    #     assert all(isinstance(i, int) and i > 0 for i in queue), "实验编号必须为正整数！"
    #     self.reading_queue = queue
        
    # def __read_exp(self, exp_no):
    #     record = ltools.get_logData(self.__plp, exp_no)
    #     assert exp_no > 0, f'训练序号需为正整数，但读取到的序号为{exp_no}'
    #     self.exp_no = int(exp_no)
    #     # 计算总共需要进行的实验组数
    #     with open(self.__hcp, 'r', encoding='utf-8') as cfg:
    #         hyper_params = json.load(cfg)
    #         n_exp = 1
    #         for v in hyper_params.values():
    #             n_exp *= len(v)
    #     # 最后一组实验的实验编号
    #     self.last_expno = self.exp_no + n_exp - 1
=======
>>>>>>> fe67262f0be35bf6395172a41ccd9efe30baa9c9

    def __iter__(self):
        if self.is_train:
            return self.__train_iter__()
        else:
            return self.__predict_iter__()

    def __train_iter__(self):
        """训练迭代
        每次提供一个实验对象，包含有单次训练的超参数组合以及动态运行参数组合，训练过程中不可改变。
        每次迭代会更新运行配置参数。
        :return: None
        """
        # 读取实验编号
        self.__read_expno()
        with open(self.__hcp, 'r', encoding='utf-8') as cfg:
            hyper_params = json.load(cfg)
            hp_keys = hyper_params.keys()
        for hps in ptools.permutation([], *hyper_params.values()):
            hyper_params = {k: v for k, v in zip(hp_keys, hps)}
            # 构造实验对象
            print(
                f'\r---------------------------'
                f'实验{self.exp_no}号/{self.last_expno}号'
                f'---------------------------'
            )
            cur_exp = New2Experiment(
                self.exp_no, self.dao_ds, self.net_type,
<<<<<<< HEAD
                hyper_params, self.cfg_dict, self.is_train
=======
                hyper_params, self.cfg_dict, self.is_train,
                save_net_fn=self.get_save_net_fn()
>>>>>>> fe67262f0be35bf6395172a41ccd9efe30baa9c9
            )
            yield cur_exp
            # 记录
            if self["log_root"] is not None:
                if hasattr(cur_exp, "metric_log"):
                    self.__write_log(self.__mlp, **cur_exp.metric_log)
                else:
                    print("没有为实验对象注入指标结果，本次实验不记录指标数据！")
                self.__write_log(self.__plp, **cur_exp.perf_log)
<<<<<<< HEAD
                if cur_exp.net is None:
                    print('训练器对象未得到训练网络对象，因此不予保存网络！')
                else:
                    self.__save_net(cur_exp.net)
=======
                # if cur_exp.net is None:
                #     print('训练器对象未得到训练网络对象，因此不予保存网络！')
                # else:
                #     self.__save_net(cur_exp.net)
>>>>>>> fe67262f0be35bf6395172a41ccd9efe30baa9c9
                self.__plot_history(cur_exp.train_mlog)
            else:
                print("没有指定保存目录，本次实验不记录结果！")
            # 更新实验编号以及读取下一次实验的参数
            self.exp_no += 1
            # 读取运行动态参数
            self.__read_runtime_cfg()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __predict_iter__(self):
        """预测迭代
        每次提供一个实验对象，包含有单次训练的超参数组合以及动态运行参数组合，训练过程中不可改变。
        每次迭代会更新运行配置参数。
        :return: None
        """
        # 读取实验编号"
        for exp_no in self.reading_queue:
            print(
                f'---------------------------实验{exp_no}号的结果'
                f'---------------------------'
            )
            # 读取实验记录
            record = ltools.get_logData(self.__mlp, exp_no)
            record["f_req_shp"] = get_req_sha(record['f_req_shp']) 
            record["l_req_shp"] = get_req_sha(record['l_req_shp'])
            # 判断是否需要加载数据集
            cur_exp = New2Experiment(
                exp_no, self.dao_ds, self.net_type,
                record, self.cfg_dict, self.is_train, 
                os.path.join(self.__np, f"{exp_no}.ptsd")
            )
            yield cur_exp
            # 保存包装好的预测结果
            if hasattr(cur_exp, "wrapped_results"):
                cur_exp_wrp = os.path.join(self.__wrp + f"{exp_no}", "") 
                os.makedirs(cur_exp_wrp, exist_ok=True)
                for i, wr in enumerate(cur_exp.wrapped_results):
                    wr.save(cur_exp_wrp + f'{i}.png')
                print(f"包装结果已保存到了{cur_exp_wrp}目录下")
            # 读取运行动态参数
            self.__read_runtime_cfg()
            print(
                f'------------------------已保存实验{exp_no}号的结果'
                '------------------------'
            )

    # def __iter__(self):
    #     """迭代
    #     每次提供一个实验对象，包含有单次训练的超参数组合以及动态运行参数组合，训练过程中不可改变。
    #     每次迭代会更新运行配置参数。
    #     :return: None
    #     """
    #     # 读取实验编号
    #     self.__read_expno()
    #     with open(self.__hcp, 'r', encoding='utf-8') as cfg:
    #         hyper_params = json.load(cfg)
    #         hp_keys = hyper_params.keys()
    #     for hps in ptools.permutation([], *hyper_params.values()):
    #         hyper_params = {k: v for k, v in zip(hp_keys, hps)}
    #         # 构造实验对象
    #         print(
    #             f'\r---------------------------'
    #             f'实验{self.exp_no}号/{self.last_expno}号'
    #             f'---------------------------'
    #         )
    #         cur_exp = New2Experiment(
    #             self.exp_no, self.dao_ds, self.net_type,
    #             hyper_params, self.cfg_dict, self.is_train
    #         )
    #         yield cur_exp
    #         # 记录
    #         if hasattr(self, "log_root"):
    #             if hasattr(cur_exp, "metric_log"):
    #                 self.__write_log(self.__mlp, **cur_exp.metric_log)
    #             else:
    #                 print("没有为实验对象注入指标结果，本次实验不记录指标数据！")
    #             self.__write_log(self.__plp, **cur_exp.perf_log)
    #             if cur_exp.net is None:
    #                 print('训练器对象未得到训练网络对象，因此不予保存网络！')
    #             else:
    #                 self.__save_net(cur_exp.net)
    #             self.__plot_history(cur_exp.train_mlog)
    #         else:
    #             print("没有指定保存目录，本次实验不记录结果！")
    #         # 更新实验编号以及读取下一次实验的参数
    #         self.exp_no += 1
    #         # 读取运行动态参数
    #         self.__read_runtime_cfg()
    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()

    def __getitem__(self, item):
        """获取控制面板中的运行配置参数。

        :param item: 运行配置参数名称
        :return: 运行配置参数值
        """
        assert item in self.cfg_dict.keys(), f'设置文件中不存在{item}参数！'
        return self.cfg_dict[item]

    def __read_runtime_cfg(self):
        """读取运行配置并更新或者赋值"""
        with open(self.__rcp, 'r', encoding='utf-8') as config:
            config_dict = jsonref.load(config, merge_props=True)
            config_dict = resolve_jsonref(config_dict)
            self.cfg_dict = config_dict

    def __write_log(self, path, **kwargs):
        """日志编写函数
        :param kwargs: 日志条目内容
        :return: None
        """
        directory, file_name = os.path.split(path)
        ltools.write_log(path, **kwargs)
        print(f'已在{path}编写日志{file_name}')

    def __save_net(self, net):
        """保存实验对象持有网络
        根据动态运行参数进行相应的网络保存动作，具有三种保存模式，保存模式由动态运行参数save_net指定：
        entire：指持久化整个网络对象
        state：指持久化网络对象参数
        no：指不进行持久化

        :return: None
        """
        if self.__np is None:
            print("未指定模型保存路径，不予保存模型！")
            return
        save_net = self.cfg_dict['save_net']
        if save_net == 'entire':
            obj_to_be_saved = net
            path = os.path.join(self.__np, f'{self.exp_no}.ptm')
        elif save_net == 'state':
            obj_to_be_saved = net.state_dict()
            path = os.path.join(self.__np, f'{self.exp_no}.ptsd')
        else:
            warnings.warn(
                f'请检查setting.json中参数save_net设置是否正确，可设置取值为：{save_net_range}'
                f'本次不予保存模型！', UserWarning
            )
            return
        torch.save(obj_to_be_saved, path)
        print(f'已在{self.__np}保存网络，保存路径为：{path}')

<<<<<<< HEAD
=======
    def get_save_net_fn(self):
        """保存实验对象持有网络
        根据动态运行参数进行相应的网络保存动作，具有三种保存模式，保存模式由动态运行参数save_net指定：
        entire：指持久化整个网络对象
        state：指持久化网络对象参数
        no：指不进行持久化

        :return: None
        """
        if self.__np is None:
            print("未指定模型保存路径，不予保存模型！")
            return
        save_net = self.cfg_dict['save_net']
        if save_net in save_net_range:
            return functools.partial(
                save_net_fn, 
                compare=self.compare, exp_no=self.exp_no, save_path=self.__np, save_format=save_net
            )
        else:
            warnings.warn(
                f'请检查setting.json中参数save_net设置是否正确，可设置取值为：{save_net_range}'
                f'本次不予保存模型！', UserWarning
            )
            return

>>>>>>> fe67262f0be35bf6395172a41ccd9efe30baa9c9
    def __plot_history(self, history) -> None:
        """绘制历史趋势图
        根据需要绘制历史趋势图，有三种模式可选，模式选择由动态运行参数plot_history指定：
        plot： 绘制图像，但不进行保存
        save：绘制图像且保存，需要指定保存路径self.__pp，保存图片名为(实验编号.jpg)
        no：不绘制历史趋势图

        :param history: 历史记录对象
        :param plot_kwargs: 历史趋势图绘制关键字参数
        :return: None
        """
        # 检查参数设置
        cfg = self.cfg_dict['plot_history']
        mute = self.cfg_dict['plot_mute']
        assert isinstance(mute, bool), "plot_mute参数需要为布尔值，请检查setting.json中参数plot_mute设置是否正确！"
        if cfg == 'no':
            return
        elif cfg == 'plot':
            savefig_as = None
        elif cfg == 'save':
            if self.__pp is None:
                warnings.warn('未指定绘图路径，不予保存历史趋势图！')
                savefig_as = None
            else:
                savefig_as = os.path.join(self.__pp, f"{self.exp_no}.png")
        else:
            raise ValueError(f'请检查setting.json中参数plot_history设置是否正确！支持的参数包括{plot_history_range}'
                             '本次不予绘制历史趋势图！')
        # 绘图
        with warnings.catch_warnings():
            # 忽视空图图例警告
            warnings.simplefilter('ignore', category=UserWarning)
            ltools.plot_history(
                history, mute=mute, title=f'EXP NO.{self.exp_no}',
                savefig_as=savefig_as, **self.plot_kwargs
            )

    def set_plot_kwargs(self, **kwargs):
        self.plot_kwargs = kwargs

    @property
    def config(self):
        return self.cfg_dict

    @property
    def device(self):
        return torch.device(self['device'])
    
    @property
    def compare(self):
        return self.__compare_fn
    
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
        assert len(pos_or_kwargs) == 2, f"比较方法接受的参数需要为两个字典对象！检测到输入的比较方法签名中，需要的位置参数数量为{len(pos_or_kwargs)}"
        self.__compare_fn = fn
