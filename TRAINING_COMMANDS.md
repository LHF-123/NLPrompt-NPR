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

直接调用适合调试单次实验或传入额外配置项：

```bash
python train.py \
  --root /path/to/datasets \
  --seed 1 \
  --trainer NLPrompt \
  --dataset-config-file configs/datasets/web_aircraft.yaml \
  --config-file configs/trainers/NLPrompt/rn50.yaml \
  --output-dir output/web_aircraft/NLPrompt/rn50_16shots/noise_real_0/lr0.001/seed1_regE0.001 \
  DATASET.NUM_SHOTS 16 \
  DATASET.NOISE_RATE 0 \
  DATASET.NOISE_TYPE real \
  DATASET.num_class 100 \
  DATASET.REG_E 0.001 \
  OPTIM.LR 0.001
```

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
