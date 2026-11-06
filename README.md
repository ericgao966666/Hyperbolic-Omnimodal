# VAST / GRAM / PMRL 的可移植 5+25 实验

本分支用于另一台服务器重跑三种原生基线：共同从VAST checkpoint初始化，前5轮只训练有效对齐head，后25轮联合训练。

**从 [portable_baselines/README.md](portable_baselines/README.md) 开始。**

迁移包包含固定训练/验证/测试manifest、独立源码、明确protocol、目标服务器路径模板、周期评估、检查点与恢复入口。媒体、模型权重、缓存和运行产物不进入Git。

仓库其余目录来自基础分支的历史交接内容；本次实验只使用 `portable_baselines/`，不执行旧入口。目标服务器须先完成配置、依赖、数据校验与真实GPU预检。
