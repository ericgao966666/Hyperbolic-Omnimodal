# VAST / GRAM / PMRL：固定15轮、无早停

本分支统一VAST初始化，执行5轮对齐head＋10轮联合训练；每轮联合验证ITC/ITM，完成全部15轮后选验证ITM最优模型做正式测试。

从 [portable_baselines/README.md](portable_baselines/README.md) 开始。旧5+25分支保留，请使用新的代码和输出目录。媒体、权重、缓存和实际运行记录不进入Git。
