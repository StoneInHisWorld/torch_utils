import random
from typing import Tuple, List

import PIL.Image
import numpy as np
from PIL import Image as IMAGE, ImageDraw
from PIL.Image import Image
from tqdm import tqdm

from utils.func.pytools import check_para


def resize_img(image: Image, required_shape: Tuple[int, int], interpolation='nearest') -> Image:
    """重塑图片

    在不改变图片的比例的情况下进行图片重塑。如改变图片比例不影响任务，请使用PIL.Image.Image.resize()方法提升效率。
    先将图片等比例放大到最大（放缩到最小）满足required_shape的某个维度，再对图片随机取部分或填充黑边以适配所需形状
    :param interpolation: 指定图片resize的插值方法
    :param image: 待编辑图片
    :param required_shape: 所需形状，(宽度，高度)
    :return: 重塑完成的图片。
    """
    # 实现将图片进行随机裁剪以达到目标shape的功能
    iw, ih = image.size
    w, h = required_shape

    # 长边放缩比例
    scale = max(h / ih, w / iw)
    # 计算新图片shape
    new_h = int(ih * scale)
    new_w = int(iw * scale)
    # 计算图片缺失shape
    dh = h - new_h
    dw = w - new_w

    # 等比例缩放数据
    def __get_interpolation():
        """获取插值方法"""
        supported = ['nearest', 'bicubic', 'bilinear', 'lanczos']
        assert check_para('interpolation', interpolation, supported)
        if interpolation == 'nearest':
            return PIL.Image.NEAREST
        elif interpolation == 'bicubic':
            return PIL.Image.BICUBIC
        elif interpolation == 'bilinear':
            return PIL.Image.BILINEAR
        elif interpolation == 'lanczos':
            return PIL.Image.LANCZOS

    image = image.resize((new_w, new_h), __get_interpolation())
    # 若需求图片大小较大，则进行填充
    if dh > 0 or dw > 0:
        back_ground = IMAGE.new(image.mode, (h, w), 0)
        back_ground.paste(image)
        image = back_ground
    # 若需求图片大小较小，则随机取部分
    if dh < 0 or dw < 0:
        i_w = random.randint(0, -dw) if dw < 0 else 0
        i_h = random.randint(0, -dh) if dh < 0 else 0
        image = image.crop((i_w, i_h, i_w + h, i_h + w))
    return image


def crop_img(img: Image, required_shape, loc: str or Tuple[int, int]) -> Image:
    """按照指定位置裁剪图片

    :param img: 即将进行裁剪的图片
    :param required_shape: 需要保留的尺寸
    :param loc: 裁剪的位置。
        可以指定为“lt, lb, rt, rb, c”的其中一种
        或者指定为二元组指示裁剪区域的左上角坐标(x, y)
    :return: 裁剪完成的图片
    """
    img_size = img.size
    iw, ih = img_size
    rw, rh = required_shape
    assert rh <= ih and rw <= iw, (
        f'裁剪尺寸{required_shape}不能超出图片尺寸{img_size}！'
    )
    if isinstance(loc, str):
        if loc == 'lt':
            loc = (0, 0, 0 + rw, 0 + rh)
        elif loc == 'lb':
            loc = (0, ih - rh, 0 + rw, iw)
        elif loc == 'rt':
            loc = (iw - rw, 0, iw, rh)
        elif loc == 'rb':
            loc = (iw - rw, ih - rh, ih, iw)
        elif loc == 'c':
            loc = (iw // 2 - rw // 2, ih // 2 - rh // 2, iw // 2 + rw // 2, ih // 2 + rh // 2)
        else:
            raise Exception(f'不支持的裁剪位置{loc}！')
    elif isinstance(loc, tuple) and len(loc) == 2:
        loc = (loc[0], loc[1], loc[0] + rw, loc[1] + rh)
    else:
        raise Exception(f'无法识别的裁剪位置参数{loc}')
    return img.crop(loc)


def read_img(path: str, mode: str = 'L'):
    """
    读取图片
    :param path: 图片所在路径
    :param mode: 图片读取模式
    :return: 图片对应numpy数组，形状为（通道，图片高，图片宽，……）
    """
    img_modes = ['L', 'RGB', '1']
    assert mode in img_modes, f'不支持的图像模式{mode}！'
    img = IMAGE.open(path).convert(mode)
    return img


def binarize_img(img: Image, threshold: int = 127) -> Image:
    """
    将图片根据阈值进行二值化
    参考自：https://www.jianshu.com/p/f6d40a73310f
    :param img: 待转换图片
    :param threshold: 二值图阈值
    :return: 转换好的图片
    """
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)
    # 图片二值化
    return img.point(table, '1')


def concat_imgs(*groups_of_imgs_labels_list: Tuple[Image, str],
                **kwargs) -> List[Image]:
    """
    拼接图片。将输入图片拼接到白板上，并附以标签和脚注（目前仅支持一张图片一行脚注），一次性生成多张结果图。
    :param groups_of_imgs_labels_list: 图片-标签序列组。包含多个序列，每个序列中包含有粘贴到单个白板上的所有图片-标签对。
    :param kwargs: 关键词参数。支持的参数包括：comment，单个白板的脚注；text_size，标签及脚注的文字大小；border_size，图片/标签/边界之间的宽度
    :return: 生成的结果图序列
    """
    footnotes = kwargs['footnotes'] if 'footnotes' in kwargs.keys() \
        else ['' for _ in range(len(groups_of_imgs_labels_list))]
    text_size = kwargs['text_size'] if 'text_size' in kwargs.keys() else 15
    border_size = kwargs['border_size'] if 'border_size' in kwargs.keys() else 5
    img_size = kwargs['img_size'] if 'img_size' in kwargs.keys() else None
    required_shape = kwargs['required_shape'] if 'required_shape' in kwargs.keys() else None
    # 判断白板所用模式
    mode = '1'
    modes = set()
    for (img, _) in groups_of_imgs_labels_list[0]:
        modes.add(img.mode)
    if 'CMYK' in modes:
        mode = 'CMYK'
        color = (255, 255, 255, 255)
        text_color = (0, 0, 0, 0)
    elif 'RGB' in modes:
        mode = 'RGB'
        color = (255, 255, 255)
        text_color = (0, 0, 0)
    elif 'F' in modes or 'L' in modes or '1' in modes:
        if 'F' in modes:
            mode = 'F'
        elif 'L' in modes:
            mode = 'L'
        else:
            mode = '1'
        color = 255
        text_color = 0
    else:
        raise NotImplementedError(f'暂未识别的图片模式组合{modes}，无法进行背景板颜色定义以及返图模式推断')

    def _impl(footnotes: str = "",
              *imgs_and_labels: Tuple[Image, str]
              ) -> Image:
        """拼接图片和标签，再添加脚注形成单张图片"""
        # 计算绘图窗格大小
        if img_size:
            cb_height, cb_width = img_size
        else:
            cb_height = max([img.height for img, _ in imgs_and_labels])
            cb_width = max([img.width for img, _ in imgs_and_labels])
        n_cube = len(imgs_and_labels)
        # 计算脚注空间
        n_ftn_lines = footnotes.count('\n') + 2  # 加上'COMMENT:'和内容
        text_indent = int(np.ceil(text_size / 3))
        ftn_height = text_size * n_ftn_lines + text_indent * (n_ftn_lines - 1)
        ftn_width = int(max([len(l) for l in footnotes.split('\n')]) * text_size / 2.5)
        # 绘制白板
        wb_width = max(
            (n_cube + 1) * border_size + n_cube * cb_width,
            2 * border_size + ftn_width
        )
        wb_height = 3 * border_size + text_size + text_indent + cb_height + ftn_height
        # 制作输入、输出、标签对照图
        whiteboard = IMAGE.new(mode, (wb_width, wb_height), color=color)
        draw = ImageDraw.Draw(whiteboard)
        # 绘制标签和图片
        for i, (img, label) in enumerate(imgs_and_labels):
            if img_size:
                img = resize_img(img, img_size)
            # 添加图片标题
            draw.text(
                ((i + 1) * border_size + i * cb_width, border_size),
                label, fill=text_color
            )
            # 拼接图片
            whiteboard.paste(
                img.convert(mode),
                (
                    (i + 1) * border_size + i * cb_width,
                    border_size + text_size + text_indent
                )
            )
        # 绘制脚注
        draw.text(
            (border_size, wb_height - ftn_height - border_size),
            'COMMENT: \n' + footnotes, fill=text_color
        )
        return whiteboard

    rets = []
    with tqdm(
            zip(groups_of_imgs_labels_list, footnotes), total=len(groups_of_imgs_labels_list),
            unit='张', position=0, desc=f"正在拼装图片中……",
            mininterval=1, leave=True, ncols=80
    ) as pbar:
        for imgs_and_labels, foot_note in pbar:
            ret = _impl(foot_note, *imgs_and_labels)
            if required_shape is not None:
                ret = ret.resize(required_shape)
            rets.append(ret)
    return rets


def get_mask(pos: List[Tuple[int, int]], size: List[int or Tuple[int]], img_channel, img_shape):
    """
    根据孔位、孔径、图片参数来获取掩膜。
    :param pos: 孔位列表，方形孔左上角位
    :param size: 孔径列表，一个孔位对应一个孔径，若孔径为整型数字，则认为长宽一致，否则需指定孔径为（长，宽）
    :param img_channel: 图片通道
    :param img_shape: 图片形状
    :return: 掩模二值图，孔径对应位置值为1，其他位置为0
    """
    mask = np.zeros((img_channel, *img_shape), dtype=int)
    for p, s in zip(pos, size):
        if type(s) == int:
            s = (s, s)
        mask[:, p[0]: p[0] + s[0], p[1]: p[1] + s[1]] = 1
    return mask


def add_mask(images: List[np.ndarray], mask: np.ndarray) -> np.ndarray:
    """
    给图片加上掩膜。
    :param images: 待操作图片。待操作图片需要转化为numpy N维数组列表，在计算过程中，会将列表转化为N维数组以加速运算。
    :param mask: 二值图掩膜
    :return: 加上掩膜的图片序列
    """
    img_shape, mask_shape = images[0].shape, mask.shape
    assert img_shape == mask_shape, f'掩膜的大小{mask_shape}须与图片的大小{img_shape}一致！'
    ret = np.array(images) * mask
    return ret
    # IMAGE.fromarray(ret[0].reshape((256, 256))).show()  # 查看掩膜图片的语句


def extract_holes(images: np.ndarray,
                  hole_poses: List[Tuple[int, int]],
                  hole_sizes: List[int]) -> list:
    """
    根据孔径大小和位置提取图片孔径内容
    :param images: 图片序列。
    :param hole_poses: 孔径位置，即方形孔径的左上角坐标。
    :param hole_sizes: 孔径大小，需为整数列表，内含孔径边长。目前只支持方形孔径。
    :return: 挖出的孔径列表。
    """
    # 检查越界问题
    assert np.max(np.array(hole_poses)[:, 0]) < images.shape[
        -2], f'孔径的横坐标取值{np.max(np.array(hole_poses)[:, 0])}越界！'
    assert np.max(np.array(hole_poses)[:, 1]) < images.shape[
        -1], f'孔径的纵坐标取值{np.max(np.array(hole_poses)[:, 1])}越界！'
    hole_sizes = np.array(hole_sizes)
    images = np.array(images)
    # 挖孔
    holes = [[] for _ in range(len(images))]
    for (x, y), size in zip(hole_poses, hole_sizes.flatten()):
        for i, image in enumerate(images[:, :, x: x + size, y: y + size]):
            holes[i].append(image)
    return holes


def extract_and_cat_holes(images: np.ndarray,
                          hole_poses: List[Tuple[int, int]],
                          hole_sizes: List[int],
                          num_rows: int, num_cols: int,
                          required_shape=None) -> np.ndarray:
    """
    根据孔径大小和位置提取图片孔径内容，并将所有孔径粘连到一起，形成孔径聚合图片。
    :param required_shape: 指定返回的图片大小
    :param images: 图片序列。
    :param hole_poses: 孔径位置，即方形孔径的左上角坐标。
    :param hole_sizes: 孔径大小，需为整数列表，内含孔径边长。目前只支持方形孔径。
    :param num_rows: 孔径矩阵的行数。
    :param num_cols: 孔径矩阵的列数。
    :return: 孔径聚合图片结果。
    """
    # 检查越界问题
    assert np.max(np.array(hole_poses)[:, 0]) < images.shape[
        -2], f'孔径的横坐标取值{np.max(np.array(hole_poses)[:, 0])}越界！'
    assert np.max(np.array(hole_poses)[:, 1]) < images.shape[
        -1], f'孔径的纵坐标取值{np.max(np.array(hole_poses)[:, 1])}越界！'
    hole_sizes = np.array(hole_sizes).reshape([num_rows, num_cols])
    # 逐行求出最大宽度作为最终的特征图片宽度
    fea_width = np.max(hole_sizes.sum(0))
    fea_height = np.max(hole_sizes.sum(1))
    whiteboard_channel = images.shape[1]
    whiteboards = np.zeros([len(images), whiteboard_channel, fea_height, fea_width])
    # 获取粘贴位置矩阵
    paste_poses = []
    for i in range(num_rows):
        for j in range(num_cols):
            x = hole_sizes[:i, j].sum()
            y = hole_sizes[i, :j].sum()
            paste_poses.append((x, y))
    images = np.array(images)
    # 逐位点粘贴
    for pp, p, size in zip(paste_poses, hole_poses,
                           hole_sizes.flatten()):
        ppx, ppy = pp
        px, py = p
        whiteboards[:, :, ppx: ppx + size, ppy: ppy + size] = images[:, :, px: px + size, py: py + size]
    # TODO：效率低下，且resize出的图片边缘不清晰
    if required_shape is not None:
        del images
        images = []
        for img in whiteboards:
            img = PIL.Image.fromarray(img.reshape(img.shape[1:]))
            img = np.array(img.resize(required_shape))
            images.append(img.reshape(whiteboard_channel, *img.shape))
        whiteboards = np.array(images)
    return whiteboards


def mean_LI_of_holes(images: np.ndarray,
                     hole_poses: List[Tuple[int, int]],
                     hole_sizes: List[int]):
    # 检查越界问题
    assert np.max(np.array(hole_poses)[:, 0]) < images.shape[
        -2], f'孔径的横坐标取值{np.max(np.array(hole_poses)[:, 0])}越界！'
    assert np.max(np.array(hole_poses)[:, 1]) < images.shape[
        -1], f'孔径的纵坐标取值{np.max(np.array(hole_poses)[:, 1])}越界！'
    for (x, y), size in zip(hole_poses, hole_sizes):
        images[:, :, x: x + size, y: y + size] = np.mean(
            images[:, :, x: x + size, y: y + size], axis=(2, 3), keepdims=True
        )
    return images


def get_mean_LI_of_holes(images: np.ndarray,
                         hole_poses: List[Tuple[int, int]],
                         hole_sizes: List[int]):
    """计算图片序列中，每个指定挖孔区域的平均光强

    :param images: 三维图片序列，每张图片要求维度信息为（通道数，长度，宽度）
    :param hole_poses: 每张图片的挖孔位置序列，该位置序列会应用到所有图片的计算中。
    :param hole_sizes: 每个挖孔位置的挖孔大小，要求len(hole_sizes) == len(hole_poses)。
    :return: 光强序列，序列中的每个元素为挖孔所得光强序列。
    """
    # 检查越界问题
    assert np.max(np.array(hole_poses)[:, 0]) < images.shape[
        -2], f'孔径的横坐标取值{np.max(np.array(hole_poses)[:, 0])}越界！'
    assert np.max(np.array(hole_poses)[:, 1]) < images.shape[
        -1], f'孔径的纵坐标取值{np.max(np.array(hole_poses)[:, 1])}越界！'
    assert len(hole_poses) == len(hole_sizes), \
        f'挖孔位置需要和挖孔大小一一对应，提供了{len(hole_poses)}个位置但只收到了{len(hole_sizes)}个大小要求。'
    mean_LIs_groups = []
    if len(images.shape) <= 3:
        # 如果只有一张图片
        images = [images]
    for image in images:
        mean_LIs = []
        for (x, y), size in zip(hole_poses, hole_sizes):
            mean_LIs.append(np.mean(image[:, x: x + size, y: y + size]))
        mean_LIs_groups.append(mean_LIs)
    return mean_LIs_groups


def blend(required_shapes: List, group_of_values: List,
          group_of_n_rows: List, group_of_n_cols: List, mode='L',
          dtype=np.float32) -> list[np.ndarray]:
    """按照给定大小以及给定颜料值生成一组晕染图

    :param required_shapes: 晕染图组中，每一张晕染图的大小
    :param group_of_values: 每一张晕染图中填充的颜料值，要求颜料块的数量和颜料值一致，即n_rows * n_cols == len(values)
    :param group_of_n_rows: 每一张晕染图中颜料的行数
    :param group_of_n_cols: 每一张晕染图中，颜料的列数
    :param mode: 晕染图的图片模式
    :return: 晕染图组列表
    """
    supported = ['L', '1', 'RGB']
    if mode == 'L' or mode == '1':
        channel = 1
    elif mode == 'RGB':
        channel = 3
    else:
        raise NotImplementedError(f'暂未支持的图片模式{mode}，支持的模式包括{supported}')
    blackboards = []
    for req_sha, values, n_rows, n_cols in zip(required_shapes, group_of_values, group_of_n_rows, group_of_n_cols):
        assert n_rows * n_cols == len(values), f'提供的颜料数量{len(values)}应与绘制区域数目{n_rows * n_cols}一致'
        blackboard = np.zeros((channel, *req_sha), dtype=dtype)
        # 绘制区域大小
        area_height = req_sha[0] // n_rows
        area_width = req_sha[1] // n_cols
        # 绘制区域坐标
        xv = [area_height * i for i in range(n_rows)]
        yv = [area_width * i for i in range(n_cols)]
        yv, xv = np.meshgrid(xv, yv)
        hole_pos = [(x, y) for x, y in zip(xv.reshape([-1]), yv.reshape([-1]))]
        # 绘制平均亮度值
        for value, (x, y) in zip(values, hole_pos):
            blackboard[:, x: x + area_height, y: y + area_width] = value
        blackboards.append(blackboard)
    return blackboards
