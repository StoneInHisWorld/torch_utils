import json


def init_log(path):
    """日志初始化方法
    日志初始化只会增加一个条目，该条目里只有“exp_no==1”的信息

    :param path: 日志所需存储路径
    """
    with open(path, 'w', encoding='utf-8') as log:
        log.write("exp_no\n1\n")


def init_settings(path):
    """运行配置文件初始化方法
    运行配置文件初始化包括所有必要参数，用户无需修改

    :param path: 配置文件存储路径
    """
    default_settings = {
        "data_portion": 1,
        "train_portion": 0.8,
        "random_seed": 42,
        "plot_mute": True,
        "plot_history": "save",
        "print_net": False,
        "save_net": "state",
        "device": "cpu",
        "lazy": False,
        "n_workers": 1,
        "pin_memory": False,
        "cuda_memrecord": False,
        "prefetch_factor": 2,
        "max_prefetch": 2,
        "share_memory": False,
        "bkg_gen": True,
        "with_hook": False,
        "hook_mute": True,
        "bulk_preprocess": True,
        "with_checkpoint": False
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
