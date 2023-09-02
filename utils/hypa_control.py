import json

from utils.tools import permutation


class ControlPanel:

    def __init__(self, hypa_config_path, runtime_config_path):
        self.__rcp = runtime_config_path
        self.__hcp = hypa_config_path
        # with open(hypa_config_path, 'r') as config:
        #     self.__hck = json.load(config).keys()
        #     for k, v in json.load(config):
        #         setattr(self, k, v)
        # with open(hypa_config_path, 'r', encoding='utf-8') as config:
        #     hypa_s_to_select = json.load(config).values()

    def __iter__(self):
        self.__read_running_config()
        # TODO：提供超参数
        with open(self.__hcp, 'r') as config:
            hyper_params = json.load(config).values()
            return permutation([], *hyper_params)

    def __enter__(self):
        with open(self.__rcp, 'r') as config:
            self.__rck = json.load(config).keys()
            for k, v in json.load(config):
                setattr(self, k, v)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __read_running_config(self):
        with open(self.__rcp, 'r') as config:
            config_dict = json.load(config)
            assert config_dict.keys() == self.__rck, '在运行期间，不允许添加新的运行设置参数！'
            for k, v in json.load(config):
                setattr(self, k, v)
            # self.data_portion, self.random_seed, self.pic_mute, self.print_net, self.device = \
            #     json.load(config).values()