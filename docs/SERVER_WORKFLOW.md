# 服务器工作与新对话接续

更新日期：2026-10-07（Asia/Shanghai）。这是交接时的状态，执行任务前应重新检查服务器。

## 当前状态

- 当前代码、配置、数据和测试已上传服务器，最初迁移的 233 个文件均通过 SHA-256 校验。
- 项目依赖已经安装；服务器完整测试：492 passed，4 warnings，138.22 秒。
- 用户已明确恢复测速并授权优先优化单个实验；完整 5000 回合正式训练尚未启动。
- 已完成 CPU/CUDA 与线程数初测、12 回合复测和两种编译候选；服务器 513 项测试与三个算法全流程短测通过。
- 已备份并同步本轮计算开销优化：HVAC 标量读取、投影中间校验、Dueling 状态特征复用、反事实冗余快照。
- 服务器由当前工作区文件迁移，包含本机未提交修改，不是旧 Git HEAD 的导出。
- 服务器项目尚未初始化 Git；GitHub 同步、自动下载结果尚未配置。测速和验证使用 screen 后台会话。
- 本轮结论与后续单实验启动方式记录在 docs/SINGLE_TRAINING_SPEED.md；新对话须先检查当前进程。
- 服务器专用 YAML 已同步本机：configs/servers/autodl_8481c/。MATD3-Q 为 CPU 4 线程，
  Dueling 为 CPU 8 线程，反事实算法为 CPU 1 线程，interop 均为 1；没有启用编译。
- 本轮测速与验证会话均已结束，239 个项目文件的本机/服务器 SHA-256 已核对一致，原始证据已下载本机备份目录。

## 连接信息

| 项目 | 当前值 |
|---|---|
| SSH 主机 | connect.westc.seetacloud.com |
| SSH 端口 | 46979 |
| SSH 用户 | root |
| 本机专用密钥路径 | C:\Users\fdd\.ssh\autodl_training_ed25519 |
| 本机项目 | C:\daima\multi_microgrid_rl_standalone |
| 服务器项目 | /root/autodl-tmp/multi_microgrid_rl_standalone |
| 服务器 Python | /root/miniconda3/bin/python |

本文仅记录密钥的文件路径，不包含私钥内容或密码。换电脑需要重新配置公钥。
实例重建、迁移或连接信息变化后，以 AutoDL 控制台的新信息为准。
本机已有一个同主机的旧 SSH 配置使用端口 56185；不要依赖该旧端口，当前命令应显式指定 46979。

在本机 PowerShell 执行：

```powershell
ssh -i C:\Users\fdd\.ssh\autodl_training_ed25519 -p 46979 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 root@connect.westc.seetacloud.com
```

非交互命令可加 `-o BatchMode=yes`，避免缺少密钥时挂在密码提示上。
SSH 非交互环境没有默认 `python` 命令，应使用上述 Python 绝对路径；交互终端可能已激活 Conda。

## 已核对的运行环境

| 项目 | 实测值 |
|---|---|
| 操作系统 | Ubuntu 22.04.5 LTS |
| CPU | Intel Xeon Platinum 8481C |
| 容器 CPU 配额 | cpu.max 为 2500000 100000，即约 25 个 CPU 的时间配额 |
| 宿主机可见逻辑 CPU | 224；不等于容器可以持续独占 224 核 |
| 容器内存上限 | 90 GiB |
| GPU 显示名称 / 显存 | NVIDIA vGPU-32GB / 32760 MiB |
| GPU 驱动 | 570.124.04 |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| PyTorch CUDA | 12.8，torch.cuda.is_available() 为 True |
| 数据盘 | /root/autodl-tmp，容量 50 GB |
| 后台会话工具 | 已有 /usr/bin/screen；tmux 当前未安装 |

早先候选截图是 EPYC 9754，实际实例使用上述 Xeon，测速以实际实例为准。
本机使用 Python 3.13.9、PyTorch 2.11.0+cu128。比较两台机器速度前应统一软件版本，
或明确标注版本差异，不把当前短测试的差异完全归因于硬件。
本机已通过测试的 Pandas 2.3.3、pytest 8.4.2、Matplotlib 3.10.6 已用于约束服务器安装。
其余完整版本清单可通过 `python -m pip freeze` 重新核对。

## 文件位置与状态证据

- 最初迁移包：服务器 `/root/autodl-tmp/multi_microgrid_rl_standalone_server.zip`。
- 最初迁移包 SHA-256：`72dd69c4449073a26536482e05bd708b05f0497c9990907e79cae35564ad1daf`。
- 本机迁移包与临时测速脚本：`C:\daima\multi_microgrid_rl_standalone_backups\server_transfer_20261007`。
- 服务器源文件清单：项目根目录 `server_source_manifest.json`；后续文档同步记录其文件哈希更新。
- 依赖约束：服务器 `/root/autodl-tmp/server_constraints.txt`。
- 已完成的安装日志：服务器 `/root/autodl-tmp/install_dependencies_pinned.log`。
- 已完成的测试日志：服务器 `/root/autodl-tmp/server_pytest.log`。
- 临时测速脚本：服务器 `/root/autodl-tmp/server_runtime_probe.py`，初测和复测证据保存在 single_speed_*_20261007 目录。
- 仓库 scripts/benchmark_training_runtime.py 已修复学习启动阈值与反事实上下文；计时记录实际更新次数。
- 优化前服务器源码备份：`/root/autodl-tmp/single_speed_before_optimization_20261007.zip`。

## 代码同步约定

目前只有手动迁移，下面是讨论中的推荐方案，尚未建立 GitHub 仓库或自动化任务。

1. 以本机现有 Git 项目为版本管理起点，保留当前未提交的物理环境与算法改动。
2. 推荐使用私有 GitHub 仓库同步源码、配置和依赖版本。用户确定仓库后再配置推送。
3. 服务器修改源码后，先查看与本机的差异、备份将受影响的文件，再同步回本机。
   未完成 Git 同步配置时，可用 SCP 下载到本机独立暂存目录再核对；不要直接覆盖整个本机项目。
4. 使用 Git 后，服务器变更应在明确的分支上提交并推送，本机拉取同一版本。
5. 每个正式实验记录所使用的代码版本和配置。运行中的实验应固定源码，后续改动使用独立版本。
6. 算法修改继续遵守 AGENTS.md；不得为加速实验修改永久 MATD3 基线、共享物理或奖励语义。
7. 本机完整备份和历史结果不需要自动上传服务器；不得覆盖或删除已接受的研究结果。

## 结果查看与下载

普通结果使用统一入口，路径为：

```text
outputs/working_runs/<算法>/runs/<运行名称>/
├── figures/       # PNG/PDF
└── artifacts/     # 模型、配置、训练/验证记录、测试轨迹、摘要
```

- 少量图片：JupyterLab 中右键文件下载。
- 整组图片：把该次运行的 figures 压缩后下载，或由 SCP 下载文件夹。
- 完整研究记录：下载整个运行目录，保留配置、模型和数据；校验文件后再确认备份完成。
- 本机下载建议使用独立目录 `C:\daima\server_results`，按算法和运行名称分开。
- 当前正式绘图入口要求 test_daily_metrics.csv 和 test_summary.json 已完成。
  中途组会曲线应从已保存的训练/验证记录另行导出，并标注截至回合、日期、种子和代码版本；
  此功能尚未单独配置。不要为了导出图片修改训练奖励、动作或经验。
- 增量训练历史可能在 training_history.jsonl 中，training_history.json 是定期快照；
  中途读取时应核对记录是否齐全，不能把旧快照当作最新进度。

## 后台运行与监控

正式长训练应使用 screen，并保存日志。此次测速和短流程验证使用独立 screen，会话结束后不表示已启动长训练。

```bash
screen -S matd3q
cd /root/autodl-tmp/multi_microgrid_rl_standalone
# 在会话中执行确认过的训练命令；运行参数须先根据测速结果确定。
```

- `Ctrl+A` 后按 `D`：离开 screen 会话，程序继续运行。
- `screen -ls`：查看后台会话；`screen -r matd3q`：重新连接对应会话。
- 项目已有 Rich 终端训练面板；日志模式可用 `--no-dashboard`，定期输出进度。
- 训练日志应使用 `python -u` 或无缓冲输出；使用 `tail -f <日志>` 查看最新输出。
- AutoDL 实例监控查看 CPU、GPU 和内存；GPU 也可使用 `nvidia-smi` 检查。
- 运行目录内 artifacts/run_status.json 和训练记录用于检查实验阶段与完成回合。
- SSH 断开与服务器关机不同：screen 中的程序能跨 SSH 断线继续运行，但服务器关机/重启会停止程序。
- 2026-10-07 实测 sshd 的 ClientAliveInterval=0、TCPKeepAlive=yes；没有发现通过该参数
  设置的固定空闲踢出时间。这不保证平台网关、网络设备或客户端不会断线。
- SSH 心跳可帮助维持空闲连接并发现失联，不能防止断网、电脑休眠或服务器关机。

参考：[AutoDL SSH](https://www.autodl.com/docs/ssh/)、
[后台会话](https://www.autodl.com/docs/daemon/)、
[下载数据](https://www.autodl.com/docs/down/)、
[OpenSSH 客户端心跳](https://man.openbsd.org/ssh_config)。

## 新对话接续提示

在同一台电脑的这个项目里开启新对话，可以提供：

> 请先读 docs/SERVER_WORKFLOW.md，通过其中的密钥连接 AutoDL，检查服务器当前状态。
> 本次需要处理的是：……。测速和训练只有在我明确要求继续时才启动。

新对话应先读本文和 AGENTS.md，核对连接、当前文件差异和是否存在运行中的训练，
然后按本次用户任务推进。本文的连接信息不代表对未来任意训练或 GitHub 发布的授权。
