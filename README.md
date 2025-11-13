
# DSAP: Enhancing Generalization in Goal-Conditioned Reinforcement Learning

This project explores the enhancement of generalization in goal-conditioned reinforcement learning by employing state abstraction and addressing confounding bias through a novel approach Deconfounded State Abstraction for Policy learning (DSAP). 

---
## Setup

### Prerequisites

- Ubuntu 20.04
- Python 3.8

### Installation

To set up the environment for the project, follow these steps:

```bash

# create and activate the conda environment
conda create -n DSAP python=3.8
conda activate DSAP

# install the necessary dependencies
cd src/
pip install -r requirement.txt
```

## Usage Instructions

To train and test agents under different settings, execute the following command:

```bash
python train_agent.py  --mode IID   --cogitation_model causal --graph collider  --noise_objects 2 --use_state_abstraction
```

### Parameter Descriptions

The main script accepts several command-line arguments for custom configuration:

- `--exp_name`: The name for the experiment run.
- `--mode`: Specifies the mode, either 'IID' for i.i.d samples or 'OOD-S' for spurious correlation.
- `--agent`: Chooses the agent type, default DSAP.
- `--cogitation_model`: Selects the model type used in DSAP, options include 'causal', 'counterfact', 'mlp', 'gnn'.
- `--env`: Sets the environment name.
- `--graph`: Chooses the type of groundtruth graph in chemistry, options are 'collider', 'chain', 'full', 'jungle'.
- `--noise_objects`: The number of objects in the environment that are considered noisy.
- `--use_state_abstraction`: A flag to determine whether state abstraction should be used.

---


### Bibtex:
```bibtex
@inproceedings{wangDSAP2026,
  title={{DSAP}: Generalizing Goal-conditioned Reinforcement Learning Agents},
  author={Yiming Wang, Kaiyan Zhao, Ming Yang, Yan Li, Furui Liu, Jiayu Chen and Leong Hou U},
  booktitle={AAAI},
  year={2026},
}
```
