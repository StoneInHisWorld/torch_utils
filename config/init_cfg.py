import json
import pandas as pd

def init_log(path):
    """日志初始化方法
    日志初始化只会增加一个条目，该条目里只有“exp_no==1”的信息

    :param path: 日志所需存储路径
    """
    log = pd.DataFrame({'exp_no': []})
    log.to_csv(path, encoding='utf-8', index=False)


def init_settings(path):
    """运行配置文件初始化方法
    运行配置文件初始化包括所有必要参数，用户无需修改

    :param path: 配置文件存储路径
    """
    # default_settings = {
    #     "data_portion": 0.1,
    #     "train_portion": 0.8,
    #     "random_seed": 42,
    #     "plot_mute": True,
    #     "plot_history": "save",
    #     "print_net": False,
    #     "save_net": "state",
    #     "device": "cpu",
    #     "lazy": False,
    #     "shuffle": True,
    #     "n_workers": 4,
    #     "pin_memory": False,
    #     "cuda_memrecord": False,
    #     "prefetch_factor": 2,
    #     "max_prefetch": 2,
    #     "share_memory": False,
    #     "bkg_gen": True,
    #     "with_hook": False,
    #     "hook_mute": True,
    #     "bulk_preprocess": True,
    #     "with_checkpoint": False,
    #     "f_req_sha": (256, 256),
    #     "l_req_sha": (256, 256),
    #     "which_dataset": "which dataset?",
    #     "log_root": "../../log",
    #     "dataset_root": "where you put your datasets"
    # }

    default_settings = {
        "random_seed": 42,
        "plot_mute": True,
        "plot_history": "save",
        "save_net": "state",
        "device": "cpu",
        "n_workers": 8,
        "cuda_memrecord": False,
        "log_root": "../../log",
        "t_kwargs": {
            "with_hook": False,
            "hook_mute": True,
            "with_checkpoint": False,
            "print_net": False,
            "device": {"$ref": "#/device"},
            "n_workers": {"$ref": "#/n_workers"}
        },
        "ds_kwargs": {
            "dataset_root": "../../datasets",
            "n_workers": {"$ref": "#/n_workers"},
            "data_portion": 0.01,
            "bulk_preprocess": False,
            "shuffle": True,
            "which_dataset": "which?",
            "f_req_sha": [128, 128],
            "l_req_sha": [128, 128],
            "lazy": False
        },
        "dl_kwargs": {
            "train_portion": 0.8, "max_prefetch": 2, "pin_memory": True,
            "prefetch_factor": None, "bkg_gen": True,
            "num_workers": {"$ref": "#/n_workers"},
            "transit_kwargs": {
                "device": {"$ref": "#/device"},
                "share_memory": False,
                "non_blocking": True
            }
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
