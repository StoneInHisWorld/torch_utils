# examples

`examples` 目录提供可直接参考的工程骨架，覆盖训练入口、推理展示、自定义数据集实现和超参数配置。建议把它当作“最小可运行模板”：先跑通，再替换为自己的数据读取逻辑和训练参数。

## 使用示例
```bash
# 1) 编辑示例数据集定义
vim examples/example_project_structure/self-defined-ds_example.py

# 2) 配置超参数
vim examples/example_project_structure/example_hp.json

# 3) 启动训练
python examples/example_project_structure/main_example.py
```
