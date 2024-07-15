# AmeliaTF

## Overview

**TODO**
AmeliaTF is a tool

## Pre-requisites

### Dataset

To run this repository, you first need to download the amelia dataset. Follow the instructions [here](https://github.com/AmeliaCMU/AmeliaScenes/DATASET.md) to download and setup the dataset.

Once downloaded, create a symbolic link into  ```datasets```:

```bash
cd datasets
ln -s /path/to/the/amelia/dataset .
```

### Installation

This repository can be installed following the instructions [here](https://github.com/AmeliaCMU/AmeliaScenes/INSTALL.md). However, we recommend to setup all of our Amelia Framework tools. You can do so following the instructions [here](https://github.com/AmeliaCMU/AmeliaScenes/INSTALL.md)

## How to use

Activate your amelia environment (**Please follow the installation instructions above**):

```bash
conda activate amelia
```

### Training Model

To run an experiment:

```bash
cd src
python train.py data=[data_config] model=[model_config]
```

### Implementation Details

For running distributed training on systems with 4090 it may be possible that the NCCL backend runs into a deadlock and causes the code to hang. To prevent this, set the following environment variable before training.

```bash
export NCCL_P2P_DISABLE=1
```

### Evaluation

## Results

### Experiments (KSEA)

| Experiment Config |       Data Config          | Marginal |  Joint  | Agent Type | Context |
| :---------------: | :------------------------: | :------: | :-----: | :--------: | :-----: |
| ```exp1.yaml```   | ```swim_ksea_traj.yaml```  | &check;  |         |            |         |
| ```exp2.yaml```   | ```swim_ksea_traj.yaml```  |          | &check; |            |         |
| ```exp3.yaml```   | ```swim_ksea_atype.yaml``` | &check;  |         | &check;    |         |
| ```exp4.yaml```   | ```swim_ksea_atype.yaml``` |          | &check; | &check;    |         |
| ```exp5.yaml```   | ```swim_ksea_map.yaml```   | &check;  |         |            | v0      |
| ```exp6.yaml```   | ```swim_ksea_map.yaml```   |          | &check; |            | v0      |
| ```exp7.yaml```   | ```swim_ksea_map.yaml```   | &check;  |         |            | v1      |
| ```exp8.yaml```   | ```swim_ksea_map.yaml```   |          | &check; |            | v1      |
| ```exp9.yaml```   | ```swim_ksea_map.yaml```   | &check;  |         |            | v2      |
| ```exp10.yaml```  | ```swim_ksea_map.yaml```   |          | &check; |            | v2      |
| ```exp11.yaml```  | ```swim_ksea_map.yaml```   |          | &check; |            | v3      |
| ```exp12.yaml```  | ```swim_ksea_map.yaml```   |          | &check; |            | v3      |


Example for experiment 1:

```bash
python src/train.py data=swim model=exp1
```

## BibTeX

If you find our work useful in your research, please cite us!

```bibtex
@article{navarro2024amelia,
  title={Amelia: A Large Model and Dataset for Airport Surface
Movement Forecasting},
  author={Navarro, Ingrid and Ortega-Kral, Pablo and Patrikar, Jay, and Haichuan, Wang and Park, Jong Hoon and Oh, Jean and Scherer, Sebastian},
  journal={arXiv preprint arXiv:2309.08889},
  year={2024}
}
```
