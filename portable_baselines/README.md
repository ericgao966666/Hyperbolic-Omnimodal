# VAST / GRAM / PMRL：共同 VAST 初始化，5 轮 head +25 轮联合训练

此目录是可独立运行的迁移包。代码、固定 manifest、protocol、检查与运行入口都在这里；不依赖仓库其他实验目录，也不依赖源服务器路径。三种方法均使用同一份 VAST `model_step_204994.pt`，不使用 GRAM/PMRL 已发布的各自训练权重。

## 1. 固定科学协议

完整配置见 [protocol.json](protocol.json)，构造配置见 [configs/model.json](configs/model.json)。

- VAST train/val/test = **82608 / 2048 / 2051**，按 source video 分组，固定样本、caption 和顺序。外部测试固定 MSRVTT 884、VATEX 1224（12240 captions）、ActivityNet 3987、DiDeMo 995。目标服务器有更多媒体也不扩充本轮集合。
- 三方法共同 VAST 初始化 SHA256：`9005d283ea0de99549efbedeb7d21c8e25bea8fa41c0f6570296b0e7a055a6c5`。公共编码器、兼容的对齐头及 ITM 沿用该权重；优化器从零开始。
- seed50、512D、global batch256、每轮322步、共30轮/9660步。前5轮1610步仅对齐头，后25轮8050步联合训练。warmup 不增加第二遍，不跳过失败更新。
- 2个中点视频帧、1个确定性音频窗、40个文本token、无字幕；预处理与源实验保持一致。
- VAST：原生 T–VA 双向对比 CE，label smoothing=.1。GRAM：原生三模态负体积双向 CE，label smoothing=.1。PMRL：原生谱内+谱间损失，tau1=.05、tau2=.1，谱间候选为全局batch。
- 联合阶段均增加 `.1 × 原生ITM三组CE`：1组正例、2组按原生相似度采样的负例。与几何方法采用的1:1均匀负例ITM有所不同。

| 阶段 | VAST | GRAM | PMRL | 编码器 / ITM |
| --- | --- | --- | --- | --- |
| epoch1–5 | t、va投影头及可学习温度 | t/v/a投影头及可学习温度 | t/v/a投影头；tau1/tau2固定 | 冻结，使用同一份VAST特征缓存；无ITM损失 |
| epoch6–30 | 保留上述参数与Adam状态 | 同左 | 同左 | 解冻有效视觉/音频/共享文本与跨模态BERT、ITM及输入投影/嵌入 |

VAST原生融合头在PMRL中用于检索读出及负例采样，但其谱对齐损失不对该头反传；不把这种参数报告成已接受warmup训练。GRAM未使用的depth head保持冻结，缺失该权重不会影响TVA。

AdamW betas=(.9,.98)，矩阵weight decay=.01，bias/向量/标量不衰减，梯度裁剪2。峰值LR：head1e-4、视觉5e-7、其他2e-5。head从第1步开始483步warmup后cosine到9660步；第6轮新解冻模块前322步warmup后按8050步联合预算cosine。FP16 AMP，保留有界同批次/RNG重试；TF32关闭。

大型manifest使用无损 `.json.gz` 压缩，读取后继续核对原始逻辑哈希；样本、caption和顺序保持不变。`manifests/index.json` 可直接查看集合规模和哈希。

## 2. 保存与评估

- epoch0记录共同初始化；**每轮验证ITC**的双向R@1/5/10和候选R@32。
- **epoch0/5/6/10/15/20/25/30增加ITM验证**；在训练所属GPU组中复用模型，保存并恢复训练RNG。
- warmup每轮保留小型head快照；joint在epoch6/10/15/20/25/30保留模型快照。
- 每轮滚动保存 `last.pt`，包含阶段、模型、Adam、AMP及全部rank RNG；支持warmup内部、5→6轮边界、联合阶段恢复。模型与优化器写入使用全局写锁。
- `BEST.json`仅引用已保存的验证节点。**正式测试固定epoch30**，不按测试成绩选模。
- 五个固定测试集保留全部指定caption；先用本方法分数选Top32，再按ITM class1−class0排序，不混合ITC分数；漏召回计miss。
- VAST/GRAM/PMRL没有原生entailment cone，不报告人为套用的cone指标。

## 3. 在另一台服务器准备

先克隆此分支，进入本目录；本仓库根目录的其他旧入口不是本次运行入口。

```bash
git clone --single-branch --branch native-baselines-vast-init-5plus25-20261106 \
  https://github.com/ericgao966666/Hyperbolic-Omnimodal.git
cd Hyperbolic-Omnimodal/portable_baselines
cp configs/paths.example.json configs/paths.local.json
```

编辑 `configs/paths.local.json`：填写已授权的 `data_root`、独立的 `output_root`、VAST checkpoint、依赖权重目录，以及五个数据集的视频/音频目录。`output_root`应为`data_root`下新建的专用子目录，不能与媒体目录重叠。该本地配置默认不进Git。

只需要一个VAST主checkpoint，另需构造资源：

```text
<assets_root>/
  clip/EVA01_CLIP_g_14_psz14_s11B.pt
  beats/BEATs_iter3_plus_AS2M.pt
  bert/bert-base-uncased/
    config.json
    model.safetensors 或 pytorch_model.bin
    vocab.txt
    tokenizer_config.json 等原目录文件
```

模型与媒体通过目标服务器已有文件或作者官方来源准备，**不在GitHub存放大权重、媒体、缓存或训练产物**。所有checkpoint与构造资产在准备时校验并记录；运行默认离线。

本机验证使用Python3.11 / PyTorch2.8.0+cu128。先确认目标GPU和驱动能使用所选wheel，再安装项目内环境；不同硬件或依赖版本必须重新执行全部检查，不能保证跨硬件逐位一致。

```bash
pwd
command -v python3
python3 --version
df -h . /你填写的数据根目录
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="$PWD/.cache/pip"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# 可选：生成PNG/PDF曲线；CSV和Markdown汇总不依赖matplotlib。
.venv/bin/python -m pip install -r requirements-plot.txt
```

至少预留约250GiB空闲（不含既有媒体，含三模型里程碑、滚动Adam恢复点、原子写入与40GiB保护余量）。实际占用以运行记录为准。CPU内存也需容纳多个完整模型的检查点加载与评估token缓存。

## 4. 校验、缓存、启动

```bash
export PYTHONNOUSERSITE=1
.venv/bin/python -B tests.py
.venv/bin/python -B prepare.py --config configs/paths.local.json
.venv/bin/python -B audit_models.py --config configs/paths.local.json
# 确认这组GPU当前可用；参数是目标服务器物理GPU编号。
.venv/bin/python -u -B cache.py --config configs/paths.local.json --devices 0,1,2,3
# 12张可用GPU时，三种方法各用4卡并行：
.venv/bin/python -B launch.py --config configs/paths.local.json \
  --groups '0,1,2,3;4,5,6,7;8,9,10,11'
# 只有8张可用GPU时用两组，第三种方法自动排队：
# --groups '0,1,2,3;4,5,6,7'
```

`prepare.py`检查全部文件存在性和源文件大小、split及caption映射、精确token数组哈希、固定媒体样本内容哈希和解码。可加 `--full-decode` 全量检查七套输入的解码，耗时更长。大小和抽样哈希通过不等于全部媒体逐字节相同；全量训练/评估仍会检查本地文件是否变化并实际解码，不静默丢样本。

manifest保留逻辑样本身份，目标端重新记录本地size/mtime；源服务器时间戳不会被当作目标文件必须一致的条件。缺文件、音频格式或内容不匹配时会明确失败，不能通过重新筛选样本绕过。

`cache.py`只构建一次所有方法共用的VAST冻结特征。`run.py`自动为每种方法执行真实global256 GPU预检，检查head更新、原生损失、原特征/缓存一致性、5→6轮交接及检查点逐位恢复；只有当前代码、环境和world size对应的预检通过才能正式训练。**复制源服务器READY文件无效。**

默认每方法4卡，每卡64。可使用8卡/每卡32，但同一次run恢复必须保持world size；不同world会独立预检。显存由配置中的cap/reserve控制并按设备总量收紧，正式batch不因显存不足自动缩小。GPU不足时先调整分组并预检。

## 5. 查看与恢复

所有产物位于配置的 `output_root`：

```text
status.json / jobs.json / launch.json   队列、PID、命令和日志路径
logs/                                 各预检/训练/评估日志
runs/<method>/training_metrics.jsonl    loss/LR/grad/AMP/耗时
runs/<method>/validation/epoch_*.json   逐轮验证
runs/<method>/snapshots/               head或模型里程碑
runs/<method>/last.pt                  完整恢复点
runs/<method>/final.pt                 固定第30轮模型
checks/                               目标GPU预检证据
evaluation/<method>/<dataset>/         固定末轮结果与分数
reports/REPORT.md                      自动汇总
reports/validation.csv / tests.csv     机器可读表格
reports/validation.png / .pdf          安装matplotlib后自动生成
```

重新运行同一条 `launch.py` 命令可恢复监督器；目录锁阻止重复监督器，存活的已登记子进程会被接管，训练恢复会检查源码、输入、环境和world size。资源暂停返回75后自动重试。普通错误保留日志并阻塞该任务，修复资源问题后可加 `--retry-failed` 重试同一配置。代码或科学配置改变应使用新输出目录，不能冒充原实验续跑。

单独运行/恢复也可：

```bash
.venv/bin/python -u -B smoke.py --config configs/paths.local.json --method vast --devices 0,1,2,3
.venv/bin/python -u -B train.py --config configs/paths.local.json --method vast --devices 0,1,2,3 --resume
.venv/bin/python -B report.py --config configs/paths.local.json
```

监督器停止时不会杀死其训练子进程，会记录状态供重启接管。不要按共享用户名批量kill进程。不要同时对同一方法手动启动训练和运行队列。

## 6. 对照解释与验证边界

本实验统一VAST初始化、输入、global batch、5+25阶段和追加步数，保留各自原生目标与ITM采样。不同方法计算量、投影头结构并不相同；几何方法还做过额外的head超参数搜索，必须单独报告这部分预算。单seed结果不能代表训练种子不确定性，当前固定视听交集也不等于论文完整榜单。

[TESTING.md](TESTING.md)记录本机实际完成的检查；[SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json)记录来源与改动；[PACKAGE_SHA256.json](PACKAGE_SHA256.json)锁定迁移包。vendor保留作者许可证。新服务器的完整GPU预检和正式训练结果只能在该服务器运行后产生。
