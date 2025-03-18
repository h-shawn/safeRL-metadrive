# safeRL-metadrive
Safe RL algorithms (PPO-Lag &amp; SAC-Lag) tested in SafeMetaDrive envs.

## 🛠️ Dependency

Install furl via:

```shell
git clone https://github.com/liuzuxin/fsrl.git
cd fsrl
pip install -e .
```

Install metadrive via:

```shell
git clone https://github.com/metadriverse/metadrive.git
cd metadrive
pip install -e .
```

## 🚀 Quick Start

Run the following command to train PPO-Lag or SAC-Lag algorithm.

```shell
python3 train_ppol.py --task SafeMetaDrive
python3 train_sacl.py --task SafeMetaDrive
```

Running the following script to evaluate the performance of your agent.

```shell
python3 eval.py --agent-name agent_ppol
python3 eval.py --agent-name agent_sacl
```
