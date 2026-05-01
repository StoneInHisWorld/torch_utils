# config

`config` 目录用于维护 torch_utils 的运行配置与环境依赖，包含配置默认结构、环境依赖描述与框架启动所需的关键参数约定。建议在项目初始化时先同步环境依赖，再开始修改训练参数。

## 环境配置示例
```bash
# 新建环境（首次）
conda env create -f config/torch_env.yml

# 已有环境：按 yml 对齐依赖（自动安装/升级/清理）
conda env update -n <env_name> -f config/torch_env.yml --prune

# 激活环境
conda activate <env_name>
```
