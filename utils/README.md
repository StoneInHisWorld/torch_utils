# utils

`utils` 模块提供训练过程通用支撑，包括实验管理、参数控制、日志、并发封装，以及图像/张量/torch 工具函数，帮助减少样板代码并提升实验可追踪性。

## 子模块说明
- `ctrl_panel.py`：控制面板，负责读取动态参数、组织超参数组合，并向训练流程提供统一配置入口。
- `experiment.py`：实验对象，负责训练过程管理（计时、显存监控、日志写入、模型持久化、历史曲线绘制）。
- `history.py`：历史记录容器，按字段存储各轮训练/验证指标。
- `process.py` / `thread.py`：并发执行封装。
- `accumulator.py`：数值累加器，汇总训练过程中的标量指标。

> 下面 `func/*` 的函数说明优先引用函数 pydoc 的开头语句；仅在缺少 pydoc 时使用简要概括（本次已补齐缺失 pydoc）。

### func/img_tools.py
- `resize_img`：重塑图片
- `crop_img`：按照指定位置裁剪图片
- `read_img`：读取图片
- `binarize_img`：将图片根据阈值进行二值化
- `concat_imgs`：拼接图片。将输入图片拼接到白板上，并附以标签和脚注（目前仅支持一张图片一行脚注），一次性生成多张结果图。
- `get_mask`：根据孔位、孔径、图片参数来获取掩膜。
- `add_mask`：给图片加上掩膜。
- `extract_holes`：根据孔径大小和位置提取图片孔径内容
- `extract_and_cat_holes`：根据孔径大小和位置提取图片孔径内容，并将所有孔径粘连到一起，形成孔径聚合图片。
- `get_mean_LI_of_holes`：计算图片序列中，每个指定挖孔区域的平均光强
- `blend`：按照给定大小以及给定颜料值生成一组晕染图

### func/log_tools.py
- `write_log`：编写运行日志。
- `plot_history`：绘制训练历史变化趋势图
- `get_logData`：通过实验编号获取实验数据。

### func/pytools.py
- `permutation`：生成超参数列表。
- `check_path`：检查指定路径。
- `check_para`：检查参数取值是否在允许范围内，不合法时发出警告。
- `multithreading_pool`：按给定并发数执行任务池并收集每个线程的返回结果。
- `multithreading_map`：将数据切分后并行映射到任务函数，并按顺序汇总结果。
- `warning_handler`：警告处理机
- `get_computer_name`：获取一个可调用对象的有意义名称
- `is_multiprocessing`：根据工作线程数判断是否启用多进程/高并发模式。
- `get_signature`：提取可调用对象的参数签名并按参数类型分类返回。

### func/tensor_tools.py
- `__ts_to_BiLevelImg`：将单张三维张量转换为二值（1-bit）PIL 图像。
- `tensor_to_img`：将四维批量张量转换为指定模式的 PIL 图像列表。
- `img_to_tensor`：将 PIL 图像序列转换为批量张量并放置到指定设备。

### func/torch_tools.py
- `get_optimizer`：根据字符串标识创建并返回对应的 PyTorch 优化器。
- `get_ls_fn`：获取损失函数。
- `init_wb`：获取初始化方法
- `get_lr_scheduler`：获取学习率规划器
- `get_activation`：获取激活函数
- `get_norm_layer`：返回一个标准化层
- `sample_wise_ls_fn`：计算每个样本的损失值的损失函数
