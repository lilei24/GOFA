# `stage3` 分支说明

## 总览

这份文档只记录 `stage3` 分支上的改动和使用方式。

这条分支的目标很单一：

- 基于原始 `origin/main`
- 不包含 `inference` 分支上的两卡模型分片逻辑
- 只尝试把原项目默认的 DeepSpeed Stage 2 切换为 DeepSpeed Stage 3


## 这条分支改了什么

当前 `stage3` 分支只改了 **1 个文件**：

1. [run_gofa.py](/home/lilei/hw/GOFA/run_gofa.py)


## 具体改动

在 [run_gofa.py](/home/lilei/hw/GOFA/run_gofa.py) 中，原始代码是：

```python
strategy = "deepspeed_stage_2" if torch.cuda.device_count() > 1 else "auto"
```

在 `stage3` 分支上改成了：

```python
strategy = "deepspeed_stage_3" if torch.cuda.device_count() > 1 else "auto"
```

也就是说：

- 单卡运行时，策略仍然是 `auto`
- 多卡运行时，策略从 `deepspeed_stage_2` 切换成 `deepspeed_stage_3`


## 改动目的

这个改动的目的不是做模型分片，也不是改 GOFA 的前向逻辑，而是：

- 尝试让 DeepSpeed 使用更激进的参数分片策略
- 观察它是否能减轻多卡运行时的显存压力

需要明确：

- `stage3` 属于 **参数分片**
- 它不是 `device_map` 那种按层切模型
- 它也不是我们在 `inference` 分支里做的两卡手工模型分片


## 使用方式

### 训练

如果你跑训练配置，例如：

- `pretrain_dev_config.yaml`
- `instruct_dev_config.yaml`

在多卡可见时，会自动走：

```python
deepspeed_stage_3
```

运行示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_gofa.py --override ./configs/instruct_dev_config.yaml
```


### 推理

如果你跑推理配置，例如：

- `inference_config.yaml`

当前代码路径里也会使用同一个 `strategy` 变量，所以多卡可见时也会走：

```python
deepspeed_stage_3
```

运行示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_gofa.py --override ./configs/inference_config.yaml
```


## 这条分支没有做的事情

这条分支 **没有** 做下面这些事情：

- 没有加入两卡模型分片
- 没有改 `GOFAMistralModel.forward()`
- 没有加入 `device0/device1/split_layer`
- 没有关闭 Lightning 的默认设备放置
- 没有绕开 `model_to_device()`
- 没有引入 `accelerator="cpu"` 这类特殊推理路径

所以这条分支应该被理解为：

**只做了最小的 DeepSpeed Stage 3 策略替换。**


## 已知风险

根据之前的实际测试经验，这个项目直接从 Stage 2 切到 Stage 3，可能出现这些问题：

1. `validation` 进度卡在 `0`
2. 程序卡在 Lightning/DeepSpeed 初始化或首个 batch 前向
3. GPU 状态异常，后续 `nvidia-smi` 显示不正常
4. 自定义 GOFA 前向与 Stage 3 的参数 gather 时机可能不完全兼容

所以这条分支的定位更像是：

- 一个最小实验分支
- 用来验证原始 GOFA 主线是否能直接承受 `deepspeed_stage_3`

而不是已经确认稳定可用的最终方案


## 和 `inference` 分支的区别

### `stage3` 分支

- 基于 `origin/main`
- 只改 DeepSpeed 策略
- 不改模型结构
- 不改前向逻辑

### `inference` 分支

- 做了两卡手工模型分片
- 改了 GOFA/Mistral/Lightning 多处代码
- 目的是实现单样本跨两卡推理

两条分支的路线不同，不要混用理解。


## 一句话总结

`stage3` 分支是一个**最小改动实验分支**：

- 从原始主线出发
- 只把 `deepspeed_stage_2` 改成 `deepspeed_stage_3`
- 用来验证 GOFA 原始代码是否能直接跑通 DeepSpeed Stage 3
