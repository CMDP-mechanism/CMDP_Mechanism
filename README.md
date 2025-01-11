# CMDP-Deep-Learning

A novel approach combining Deep Learning with Differential Privacy using Coupled Multiplicative Differentially Private (CMDP) mechanism.

## Overview

This project introduces a novel Coupled Multiplicative Differentially Private (CMDP) mechanism integrated into the Stochastic Gradient Descent (SGD) algorithm. Unlike traditional DPSGD approaches, our method addresses key challenges including large noise magnitudes, unbounded perturbation outputs, and accumulated privacy risks.

Key features:
- Bounded and unbiased perturbations with lower noise magnitudes
- Strict ε-DP compliance
- Adaptive hyper-parameter optimization strategy
- Superior performance compared to state-of-the-art baselines

## Installation

1. Clone the repository:
```bash
git clone https://github.com/CMDP-mechanism/CMDP_Mechanism.git
cd CMDP_Mechanism
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Datasets

The following datasets are supported and included in the `data` folder:
- MNIST
- Fashion MNIST (FMNIST)
- CIFAR-10
- IMDB

Note: The IMDB dataset starts with a baseline accuracy of 50%.

## Usage

### DPSGD-CMDP Algorithm

Here are example commands for running the DPSGD-CMDP algorithm on different datasets:

#### MNIST
```bash
python3 main.py --algorithm DPSGD-CMDP --dataset_name MNIST --lr 1 --C_t 0.2 --batch_size 1024 --epsilon 5.0 --lr_T_gain 1.02 --lr_min 0.5 --lr_max 2.0 --epsilon_T_acc 80 --device cuda
```

#### Fashion MNIST
```bash
python3 main.py --algorithm DPSGD-CMDP --dataset_name FMNIST --lr 1 --C_t 0.2 --batch_size 1024 --epsilon 5.0 --lr_T_gain 1.02 --lr_min 0.5 --lr_max 2.0 --epsilon_T_acc 60 --device cuda
```

#### CIFAR-10
```bash
python3 main.py --algorithm DPSGD-CMDP --dataset_name CIFAR-10 --lr 1 --C_t 0.2 --batch_size 4096 --epsilon 5.0 --lr_T_gain 1.02 --lr_min 0.5 --lr_max 2.0 --epsilon_T_acc 30 --device cuda
```

#### IMDB
```bash
python3 main.py --algorithm DPSGD-CMDP --dataset_name IMDB --lr 0.02 --C_t 0.2 --batch_size 4096 --epsilon 5.0 --lr_T_gain 1.02 --lr_min 0.02 --lr_max 0.03 --epsilon_T_acc 55 --device cuda
```

### Baseline Algorithms

#### DPSGD
Example commands for running the DPSGD baseline:

```bash
# MNIST
python3 main.py --algorithm DPSGD --dataset_name MNIST --sigma_t 1.25 --lr 2 --C_t 0.2 --batch_size 512 --epsilon 1.0 --device cuda

# Fashion MNIST
python3 main.py --algorithm DPSGD --dataset_name FMNIST --sigma_t 1.25 --lr 2 --C_t 0.2 --batch_size 512 --epsilon 1.0 --device cuda

# IMDB (requires lower learning rate)
python3 main.py --algorithm DPSGD --dataset_name IMDB --sigma_t 1.5 --lr 0.02 --C_t 0.2 --batch_size 512 --epsilon 1.0 --device cuda
```

Note: DPSGD-HF does not support the IMDB dataset.

#### DPSGD-HF
Additional parameters required:
- `--input_norm`
- `--bn_noise_multiplier`
- `--use_scattering`

#### DPSUR
Requires all DPSGD-HF parameters plus:
- `--sigma_v`
- `--bs_valid`
- `--beta`

### Additional Features

#### Membership Inference Attack
To enable Membership Inference Attack testing, add the `--MIA True` flag to any command:
```bash
python3 main.py --algorithm DPSGD-CMDP --dataset_name MNIST --lr 1 --C_t 0.2 --batch_size 1024 --epsilon 5.0 --MIA True --device cuda
```

#### CPU Training
For systems without GPU support, replace `--device cuda` with `--device cpu`:
```bash
python3 main.py --algorithm DPSGD-CMDP --dataset_name MNIST --lr 1 --C_t 0.2 --batch_size 1024 --epsilon 5.0 --device cpu
```

## Contributions

Following the Gaussian mechanism's user-friendly APIs and cross-platform libraries, we aim to release CMDP as an open-source toolkit for seamless workflow integration to foster further research in privacy-preserving deep learning.
