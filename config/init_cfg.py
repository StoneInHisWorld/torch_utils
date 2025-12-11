import json
import pandas as pd

def init_log(path):
    """日志初始化方法
    日志初始化只会增加一个条目，该条目里只有“exp_no==1”的信息

    :param path: 日志所需存储路径
    """
    log = pd.DataFrame({'exp_no': []})
    log.to_csv(path, encoding='utf-8', index=False)


def init_train_settings(path):
    """运行配置文件初始化方法
    运行配置文件初始化包括所有必要参数，用户无需修改

    :param path: 配置文件存储路径
    """
    default_settings = {
        "random_seed": 42,
        "plot_mute": True,
        "plot_history": "save",
        "save_net": "state",
        "device": "cpu",
        "n_workers": 8,
        "cuda_memrecord": False,
        "log_root": "./log",
        "nb_kwargs": {
            "with_checkpoint": False, "print_net": False
        },
        "t_kwargs": {
            "with_hook": False, "hook_mute": True, "train_prefetch": 2, "valid_prefetch": 2,
            "tdata_q_len": 4, "vdata_q_len": 4, "device": {"$ref": "#/device"}, "n_workers": 5,
            "pbar_verbose": True
        },
        "ds_kwargs": {
            "dataset_root": ".", "n_workers": 8,
            "data_portion": 0.1, "bulk_preprocess": False, "shuffle": True,
            "device": {"$ref": "#/device"}, "share_memory": True,
            "non_blocking": True, "transit_kwargs": {}, "f_req_shp": [128, 128],
            "l_req_shp": [128, 128], "f_lazy": True, "l_lazy": True,
            "bulk_transit": False
        },
        "dl_kwargs": {
            "train_portion": 0.8, "pin_memory": False, "prefetch_factor": 2, "num_workers": 4
        }
    }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(default_settings, f)


def init_predict_settings(path):
    """运行配置文件初始化方法
    运行配置文件初始化包括所有必要参数，用户无需修改

    :param path: 配置文件存储路径
    """
    default_settings = {
        "log_root": "./log",
        "ds_kwargs": {
            "data_portion": 0.01,
            "dataset_root": "./datasets",
            "n_workers": 8,
            "bulk_preprocess": True,
            "shuffle": True,
            "f_lazy": False,
            "l_lazy": False,
            "device": "cpu",
            "non_blocking": True,
            "share_memory": False,
            "transit_kwargs": {},
            "bulk_transit": True
        },
        "dl_kwargs": {
            "batch_size": 4
        },
        "nb_kwargs": {},
        "t_kwargs": {
            "pbar_verbose": True
        }
    }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(default_settings, f)


def init_hps(path):
    """超参数文件初始化方法
    超参数文件初始化包括常用需要调整的超参数，用户可在创建后，添加自己所需超参数。
    该修改会在下次运行生效。
    每个超参数需要以列表形式赋值，以方便一键调参

    :param path: 超参数文件存储路径
    """
    default_hps = {
        "k": [10],
        "epochs": [100],
        "batch_size": [8],
        "ls_fn": ["mse"],
        "lr": [5e-5],
        "optim_str": ["adam"],
        "w_decay": [0.0],
        "init_meth": ["xavier"],
        "comment": ["默认调参"]
    }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(default_hps, f, ensure_ascii=False)
