# 训练与评测命令

本文档汇总本仓库常用训练、评测、续跑和结果解析命令。

## 环境准备

```bash
conda create -y -n nlprompt python=3.8
conda activate nlprompt
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
cd Dassl.pytorch
pip install -r requirements.txt
python setup.py develop
cd ..
```

## 使用脚本训练

脚本入口：

```bash
bash scripts/nlprompt/main.sh <DATASET> <SHOTS> <RATE> <TYPE> <CLASS>
```

示例：Caltech101 合成对称噪声训练。

```bash
DATA=/path/to/datasets \
SEED_LIST="1" \
REG_E_LIST="0.001" \
LR_LIST="0.001" \
bash scripts/nlprompt/main.sh caltech101 16 0.5 sym 100
```

WebFG-496 示例。WebFG 配置默认关闭额外合成噪声，`RATE` 和 `TYPE` 主要用于输出目录命名及兼容脚本参数。

```bash
DATA=/path/to/datasets \
SEED_LIST="1" \
REG_E_LIST="0.001" \
LR_LIST="0.001" \
bash scripts/nlprompt/main.sh web_aircraft 16 0 real 100
```

类别数对应关系：

- `web_aircraft`: `100`
- `web_bird`: `200`
- `web_car`: `196`

## 直接调用 train.py

直接调用适合调试单次实验，或者显式覆盖 batch size、学习率、epoch 等配置。

### `--trainer` 和 `--config-file` 的区别

- `--trainer NLPrompt`：选择实际执行训练逻辑的 Python Trainer 类，即 `trainers/nlprompt.py` 中注册的 NLPrompt 训练器。
- `--config-file configs/trainers/NLPrompt/rn50.yaml`：加载训练超参数配置，包括 backbone、输入尺寸、数据增强、batch size、优化器、学习率调度和 epoch 等。
- `--dataset-config-file configs/datasets/web_aircraft.yaml`：加载数据集配置，包括数据集名称、类别数、是否启用合成噪声等。

简单理解：

```text
--trainer       决定“用哪段训练代码”
--config-file   决定“这段训练代码用什么模型和超参数”
--dataset-config-file 决定“读哪个数据集”
```

### 最小训练命令

如果接受默认值，可以只写下面这些。默认 `--trainer` 是 `NLPrompt`，默认 `--seed` 是 `1`；`NUM_SHOTS` 默认是 `16`，学习率使用 `rn50.yaml` 中的 `0.002`。

```bash
python train.py \
  --root /path/to/datasets \
  --dataset-config-file configs/datasets/web_aircraft.yaml \
  --config-file configs/trainers/NLPrompt/rn50.yaml \
  --output-dir output/web_aircraft_min
```

这个命令会使用：

- 数据集：`web_aircraft.yaml` 中的 `WebAircraft`
- 模型配置：`rn50.yaml`
- Trainer：命令行默认值 `NLPrompt`
- 训练 batch size：`rn50.yaml` 中的 `DATALOADER.TRAIN_X.BATCH_SIZE: 32`
- 测试 batch size：`rn50.yaml` 中的 `DATALOADER.TEST.BATCH_SIZE: 100`

### 完整训练命令

如果希望实验完全显式，推荐写成下面这样：

```bash
python train.py \
  --root /path/to/datasets \
  --seed 1 \
  --trainer NLPrompt \
  --dataset-config-file configs/datasets/web_aircraft.yaml \
  --config-file configs/trainers/NLPrompt/rn50.yaml \
  --output-dir output/web_aircraft/NLPrompt/rn50_16shots/noise_real_0/lr0.001/seed1_regE0.001 \
  DATASET.NUM_SHOTS 16 \
  DATASET.NOISE_LABEL False \
  DATASET.NOISE_RATE 0 \
  DATASET.NOISE_TYPE real \
  DATASET.num_class 100 \
  DATASET.REG_E 0.001 \
  OPTIM.LR 0.001 \
  OPTIM.MAX_EPOCH 200 \
  DATALOADER.TRAIN_X.BATCH_SIZE 32 \
  DATALOADER.TEST.BATCH_SIZE 100 \
  DATALOADER.NUM_WORKERS 4
```

参数含义：

- `--root`：数据集根目录。例如 WebFG 应放在 `/path/to/datasets/web-aircraft/`。
- `--seed`：随机种子。相同数据、配置和环境下用于复现实验。
- `--trainer`：训练器名称。这里使用 `NLPrompt`。
- `--dataset-config-file`：数据集 YAML。WebFG-aircraft 使用 `configs/datasets/web_aircraft.yaml`。
- `--config-file`：训练配置 YAML。这里使用 RN50 版本的 `configs/trainers/NLPrompt/rn50.yaml`。
- `--output-dir`：日志、checkpoint、tensorboard 等输出目录。续跑也会优先检查这个目录。
- `DATASET.NUM_SHOTS`：每类训练样本数。`16` 表示 16-shot。
- `DATASET.NOISE_LABEL`：是否额外制造合成噪声。WebFG 是真实 web 噪声，通常设为 `False`。
- `DATASET.NOISE_RATE`：合成噪声比例。只有 `DATASET.NOISE_LABEL True` 时才真正生效。
- `DATASET.NOISE_TYPE`：合成噪声类型，如 `sym`、`asym`。WebFG 场景可写 `real` 作为记录。
- `DATASET.num_class`：类别数。`web_aircraft=100`，`web_bird=200`，`web_car=196`。
- `DATASET.REG_E`：OT 伪标签中的熵正则参数。
- `OPTIM.LR`：学习率，会覆盖 `rn50.yaml` 中的默认值。
- `OPTIM.MAX_EPOCH`：训练 epoch 数。
- `DATALOADER.TRAIN_X.BATCH_SIZE`：训练 batch size。
- `DATALOADER.TEST.BATCH_SIZE`：验证和测试 batch size。
- `DATALOADER.NUM_WORKERS`：DataLoader 进程数。Windows 或调试时可设 `0`，Linux 训练时可设 `4` 或更高。

## 续跑训练

有续跑功能。训练开始前，Trainer 会检查 checkpoint：

- 默认检查当前 `--output-dir`。
- 如果传入 `--resume <DIR>`，则从 `<DIR>` 恢复。
- NLPrompt 的 checkpoint 保存在输出目录下的 `prompt_learner/` 子目录。

因此，中断后重复运行同一个训练命令，通常会自动从当前输出目录续跑：

```bash
DATA=/path/to/datasets SEED_LIST="1" REG_E_LIST="0.001" LR_LIST="0.001" \
bash scripts/nlprompt/main.sh web_aircraft 16 0 real 100
```

如果要从指定目录续跑到新的输出目录，使用：

```bash
python train.py \
  --root /path/to/datasets \
  --resume output/web_aircraft/NLPrompt/rn50_16shots/noise_real_0/lr0.001/seed1_regE0.001 \
  --output-dir output/web_aircraft_resume \
  --trainer NLPrompt \
  --dataset-config-file configs/datasets/web_aircraft.yaml \
  --config-file configs/trainers/NLPrompt/rn50.yaml \
  DATASET.NUM_SHOTS 16 DATASET.num_class 100
```

## 评测

使用已有脚本：

```bash
bash scripts/nlprompt/eval.sh caltech101 rn50
```

或直接评测指定 checkpoint：

```bash
python train.py \
  --root /path/to/datasets \
  --trainer NLPrompt \
  --dataset-config-file configs/datasets/web_aircraft.yaml \
  --config-file configs/trainers/NLPrompt/rn50.yaml \
  --model-dir output/web_aircraft/NLPrompt/rn50_16shots/noise_real_0/lr0.001/seed1_regE0.001 \
  --load-epoch 50 \
  --eval-only \
  DATASET.NUM_SHOTS 16 DATASET.num_class 100
```

## 结果解析

```bash
python parse_test_res.py output/web_aircraft/NLPrompt/rn50_16shots/noise_real_0/lr0.001
```
