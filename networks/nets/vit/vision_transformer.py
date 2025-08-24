from collections import OrderedDict

from torch import nn

from networks.basic_nn import BasicNN
from networks.nets.resnet import ResNet50
from networks.nets.vit import supported_models
from networks.nets.vit.dpdc import EncoderLayer, FetchClass


class VisionTransformer(BasicNN):
    """
    VisionTransformer标准实现
    根据-An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale-
    https://doi.org/10.48550/arXiv.2010.11929 提供的架构编写
    """

    def __init__(self, which, in_channels, num_classes: int = None, **kwargs):
        """
        VisionTransformer标准实现

        :param which: 指定ViT的类型
        :param in_channels: 指定ViT输入数据的通道数
        :param num_classes: 指定输出向量通道数。若不指定则输出representation_size或hidden_size
        :param kwargs: BasicNN基本参数。
            在此处可以指定输入张量的形状input_size（长和宽），如不指定则默认为(224, 224)
        """
        # len_pic_seq：图片块序列长度
        # num_hiddens：词汇表中每个词汇长度
        # 将图片切块
        self.input_size = (in_channels, 224, 224) if 'input_size' not in kwargs.keys() \
            else kwargs.pop('input_size')
        self.__get_model_config(which)
        # 获取ResNet层
        resnet, output_channels = self.find_a_resnet(in_channels)
        linear_projector = nn.Conv2d(
            output_channels, self.hidden_size,
            kernel_size=self.patches_size, stride=self.patches_size
        )
        # 获取编码器
        encoder = EncoderLayer(
            self.classifier, self.hidden_size, self.tran_emb_size,
            self.tran_num_heads, self.tran_mlp_dim, self.tran_num_layers,
            self.tran_dropout_rate
        )
        fetch_class = FetchClass(self.classifier)
        # ？
        pre_logits = nn.Identity() if self.representation_size is None else \
            nn.Sequential(
                nn.Linear(self.hidden_size, self.representation_size), nn.Tanh()
            )
        # 感知机头
        if num_classes:
            mlp_head = nn.Linear(
                self.hidden_size if self.representation_size is None
                else self.representation_size,
                num_classes
            )
        else:
            mlp_head = nn.Identity()
        kwargs = {'input_size': self.input_size, **kwargs}
        super(VisionTransformer, self).__init__(OrderedDict([
            ('resnet', resnet), ('linear_projector', linear_projector),
            ('encoder', encoder), ('fetch_class', fetch_class),
            ('pre_logits', pre_logits), ('mlp_head', mlp_head)
        ]), **kwargs)

    def find_a_resnet(self, in_channels):
        """根据参数指定来分配ResNet块

        :param in_channels: 指定ResNet的输入通道数
        :return: ResNet网络，输出通道数
        """
        if self.resnet_type == 'resnet50':
            resnet = ResNet50(in_channels, [3, 4, 9])
            del resnet.avgpool, resnet.flatten, resnet.output
            output_channels = resnet.in_channels
        else:
            resnet = nn.Identity()
            output_channels = in_channels
        return resnet, output_channels

    def __get_model_config(self, which):
        assert which in supported_models, f'不支持的模型类型{which}，支持的模型类型包括{supported_models}'
        if which == 'b16':
            self.__get_b16_config()
        elif which == 'ti16':
            self.__get_ti16_config()
        elif which == 's16':
            self.__get_s16_config()
        elif which == 'l16':
            self.__get_l16_config()
        elif which == 'h14':
            self.__get_h14_config()
        elif which == 'r50_b16':
            self.__get_r50_b16_config()
        else:
            pass

    def __get_ti16_config(self):
        """指定ViT-Ti/16的配置"""
        self.resnet_type = None
        self.patches_size = 16
        self.hidden_size = 192
        self.tran_emb_size = (self.input_size[1] // self.patches_size) * (self.input_size[2] // self.patches_size)
        self.tran_mlp_dim = 768
        self.tran_num_heads = 3
        self.tran_num_layers = 12
        self.tran_attention_dropout_rate = 0.0
        self.tran_dropout_rate = 0.0
        self.classifier = 'token'
        self.representation_size = None

    def __get_s16_config(self):
        """指定ViT-S/16的配置"""
        self.resnet_type = None
        self.patches_size = 16
        self.hidden_size = 384
        self.tran_emb_size = (self.input_size[1] // self.patches_size) * (self.input_size[2] // self.patches_size)
        self.tran_mlp_dim = 1536
        self.tran_num_heads = 6
        self.tran_num_layers = 12
        self.tran_attention_dropout_rate = 0.0
        self.tran_dropout_rate = 0.0
        self.classifier = 'token'
        self.representation_size = None

    def __get_b16_config(self):
        """指定ViT_B16的配置"""
        self.resnet_type = None
        self.patches_size = 16
        self.hidden_size = 768
        self.tran_emb_size = (self.input_size[1] // self.patches_size) * (self.input_size[2] // self.patches_size)
        self.tran_mlp_dim = 3072
        self.tran_num_heads = 12
        self.tran_num_layers = 12
        self.tran_attention_dropout_rate = 0.0
        self.tran_dropout_rate = 0.0
        self.classifier = 'token'
        self.representation_size = None

    def __get_l16_config(self):
        """指定ViT_L16的配置"""
        self.resnet_type = None
        self.patches_size = 16
        self.hidden_size = 1024
        self.tran_emb_size = (self.input_size[1] // self.patches_size) * (self.input_size[2] // self.patches_size)
        self.tran_mlp_dim = 4096
        self.tran_num_heads = 16
        self.tran_num_layers = 24
        self.tran_attention_dropout_rate = 0.0
        self.tran_dropout_rate = 0.1
        self.classifier = 'token'
        self.representation_size = None

    def __get_h14_config(self):
        """指定ViT_h14的配置"""
        self.resnet_type = None
        self.patches_size = 14
        self.hidden_size = 1280
        self.tran_emb_size = (self.input_size[1] // self.patches_size) * (self.input_size[2] // self.patches_size)
        self.tran_mlp_dim = 5120
        self.tran_num_heads = 16
        self.tran_num_layers = 32
        self.tran_attention_dropout_rate = 0.0
        self.tran_dropout_rate = 0.1
        self.classifier = 'token'
        self.representation_size = None

    def __get_r50_b16_config(self):
        """Returns the Resnet50 + ViT-B/16 configuration."""
        self.__get_b16_config()
        self.resnet_type = 'resnet50'
        self.tran_attention_dropout_rate = 0.1
        self.patches_size = 1
        self.resnet_num_layers = (3, 4, 9)
        self.resnet_width_factor = 1


# inputs = torch.rand([4, 3, 224, 224])
# vit = VisionTransformer('h14', 3, num_classes=784)
# inputs = vit(inputs)
# print(inputs.shape)
