# Three-Intersection Traffic Signal Control with Reinforcement Learning in SUMO

Code used to compare single-agent PPO, parameter-shared multi-agent PPO, reward shaping, neighbour observations, ETA-based green-wave control, and tunable multi-objective reinforcement learning on a synthetic three-intersection SUMO corridor.

## Repository structure

```text
envs/        Shared SUMO network, routes, configurations, and traffic generator
SINGLE/      S01-S04 centralised single-agent experiments
MARL/        M01-M06 parameter-shared multi-agent experiments
EVALUATION/  Matched-seed green-wave time-space evaluation
```

Each experiment contains a self-contained `train.py`, `test.py`, and `plot.py`. Generated models and logs stay inside that experiment directory and are excluded from Git.

## Experiments

| ID | Controller |
| --- | --- |
| S01 | Single-agent queue reward, fixed learning rate |
| S02 | Single-agent queue reward, linearly decaying learning rate |
| S03 | Single-agent active-phase PBRS, fixed learning rate |
| S04 | Single-agent active-phase PBRS, linearly decaying learning rate |
| M01 | MARL queue reward, fixed learning rate |
| M02 | MARL queue reward, linearly decaying learning rate |
| M03A | MARL active-phase PBRS, fixed learning rate |
| M03T | MARL total-queue PBRS, fixed learning rate |
| M04 | M03A with neighbour queue observations |
| M05 | Bidirectional ETA green-wave controller |
| M06 | Preference-conditioned multi-objective controller |

## Setup

The formal experiments used Python 3.9 and Eclipse SUMO 1.24.0. Install SUMO separately, make its binaries available on `PATH`, and set `SUMO_HOME` to the SUMO installation directory. Then install the Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run an experiment

Run commands from the selected experiment directory so that `test.py` can import its local `train.py`.

```powershell
cd MARL\M04_PBRS_Communication_FixLR
$env:JUC_RUN_IDX = "1"
$env:JUC_SEED = "666"
$env:JUC_TOTAL_TIMESTEPS = "300000"
python train.py
python test.py
python plot.py
```

Use training seeds `666`, `888`, and `999` with run indices `1`, `2`, and `3` to reproduce the three formal runs. Set `JUC_USE_GUI=1` before `test.py` to show SUMO-GUI. Training defaults remain encoded in each experiment.

## Time-space evaluation

After M04, M05, and M06 models exist under their local `models/` directories:

```powershell
cd EVALUATION\GreenWave_TimeSpace
$env:JUC_RUN_IDX = "1"
$env:JUC_EVAL_SEED = "8848"
python test.py
python plot.py
```

The repository includes the three formal trained runs, their checkpoints, TensorBoard logs, SUMO episode CSV files, and test logs. New generated outputs remain ignored unless they are added explicitly.
