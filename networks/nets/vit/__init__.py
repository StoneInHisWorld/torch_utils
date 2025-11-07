supported_models = ['r50_b16', 'b16', "ti16", "s16", "l16", "h14"]
supported_classifier = ['token', 'token_unpooled', 'gap', 'unpooled']
supported_resnets = ['resnet50', None]

from .vision_transformer import VisionTransformer as ViT