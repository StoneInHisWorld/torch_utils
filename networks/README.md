# networks

`networks` 模块提供模型构建与训练编排能力，围绕 `BasicNN`、`Trainer`、`net_builder` 与 `nets/layers` 组织，支持从网络定义、训练执行到模型保存的完整流程。

## 子模块说明
- `basic_nn.py`：基础网络能力，封装优化器/损失函数/调度器准备与前后向流程。
- `trainer/`：训练流程实现，包括训练、验证、日志、hook 与 profiler 支持。
- `net_builder.py`：网络构建与初始化流程封装。
- `net_saver.py`：模型持久化与保存策略。

## nets 目录说明
- `pix2pix/`：pix2pix 对抗网络相关生成器、判别器与组合实现。参考：Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks*, CVPR 2017, DOI: `10.1109/CVPR.2017.632`.
- `unet/`：U-Net 变体（如 `u128`、`u256`）实现。参考：Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015, DOI: `10.1007/978-3-319-24574-4_28`.
- `vit/`：Vision Transformer 与相关组件。参考：Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021.
- `alexnet.py`：AlexNet 结构。参考：Krizhevsky et al., *ImageNet Classification with Deep Convolutional Neural Networks*, NeurIPS 2012.
- `googlenet.py`：GoogLeNet/Inception v1 结构。参考：Szegedy et al., *Going Deeper with Convolutions*, CVPR 2015, DOI: `10.1109/CVPR.2015.7298594`.
- `lenet.py`：LeNet 结构。参考：LeCun et al., *Gradient-Based Learning Applied to Document Recognition*, Proceedings of the IEEE 1998, DOI: `10.1109/5.726791`.
- `mlp.py` / `slp.py`：多层/单层感知机基础结构（经典前馈神经网络范式）。
- `vgg.py`：VGG 结构。参考：Simonyan & Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition*, ICLR 2015.
- `resnet.py`：ResNet 结构。参考：He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016, DOI: `10.1109/CVPR.2016.90`.
- `encoder_decoder.py`：通用编码器-解码器范式实现，可用于重建/翻译类任务。
- `elnn/`、`adaunet.py`、`itransformer.py`、`dynamic_blender.py`、`adawzynet.py`、`wzynet_essay.py`：项目内定制网络实现。

## layers 目录说明
- 结构与形状处理：`reshape.py`、`identity.py`、`multi_output.py`、`val2img.py`。
- 注意力与 Transformer 组件：`kvcache_MultiheadAttention.py`、`kvcache_Transformer.py`、`add_positionEmbeddings.py`、`patching.py`。
- 视觉网络构件：`inception.py`、`resnet_blocks.py`。
- 损失与指标相关层：`ganloss.py`、`pcc.py`、`ssim.py`、`pytorch_ssim.py`。
