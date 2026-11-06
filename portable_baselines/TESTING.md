# 验证记录

代码整理服务器使用项目已有Python3.11 / torch2.8.0+cu128环境，检查在独立工作树进行，原几何实验继续运行。

- 已实际构造VAST、GRAM、PMRL并加载同一份VAST checkpoint。所有已加载公共参数的摘要均为 `94cd4d6a82cb0bd2e05c609ddfda6fcdd72ba72fb51e96c8585d0a0d1b58b963`。
- GRAM额外缺失的depth head未参与TVA任务并保持冻结；视觉编码器未使用的logit_scale命名差异被显式记录。其余不匹配会拒绝启动。
- CPU回归已通过：七套逻辑manifest及caption映射、source-video不交叉、world1/2/4/8全局batch顺序一致、head Adam状态交接、新解冻参数无旧状态、多caption召回及TopK漏召回分母。
- 双进程Gloo对照已通过：三方法warmup早返回的对齐损失值及梯度，与原生联合前向中的对齐项逐位相同。
- 目标端prepare流程已在现有媒体上实际执行：全部路径/大小检查、重建token与原数组SHA256完全一致、固定媒体样本内容哈希通过、七套输入各4条解码通过。未做全量媒体逐字节对照。
- 已在单GPU以global batch2做三种真实模型功能检查：2次head更新、3次joint更新、重新构造模型/Adam后重放更新逐位一致，16条验证样本的ITC及ITM流程通过。该小批次检查不替代正式global256多卡预检。
- 本机未执行这套新实验的正式训练，也未声称已验证目标服务器的CUDA数值/显存。`smoke.py`是目标端正式启动前的强制门槛。
