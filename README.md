# AmeliaTF

This repository contains the model implementation, as well as the training and evaluation code of our paper:

#### Amelia: A Large Dataset and Model for Airport Surface Movement Forecasting [[paper](https://arxiv.org/pdf/2407.21185)]

[Ingrid Navarro](https://navars.xyz) *, [Pablo Ortega-Kral](https://paok-2001.github.io) *, [Jay Patrikar](https://www.jaypatrikar.me) *, Haichuan Wang,
Zelin Ye, Jong Hoon Park, [Jean Oh](https://cmubig.github.io/team/jean_oh/) and [Sebastian Scherer](https://theairlab.org/team/sebastian/)

<p align="center">
  <img width="1000" src="./assets/ksfo_results.gif" alt="Amelia">
</p>

## Overview

**AmeliaTF** is a large transformer-based trajectory forecasting model that aims to characterize relevant **airport surface movement** operations from the [Amelia-48](https://ameliacmu.github.io/amelia-dataset/) dataset.

To do so, our model comprises three main submodules:

1. A **scene representation** module that determines the agents of interest in the scene using a scoring strategy, and encodes per-agent features,
2. A transformer-based **scene encoder**, which hierarchically encodes the temporal, agent-to-agent and agent-to-context relationships within a scene, and;
3. A **trajectory decoder** that models the set of possible futures with associated confidence scores using a Gaussian Mixture Model.

<p align="center">
  <img width="1000" src="./assets/model.png" alt="Amelia">
</p>

We explore different scene representation and training experiments for our model varying from **single-airport** to **multi-airport** settings in which we assess our model’s generalization capabilities. In the subsequent sections we provide details on how to reproduce our experiments. For further details, please check out our paper!

## Pre-requisites

### Dataset

To run this repository, you first need to download the amelia dataset. Follow the instructions [here](https://ameliacmu.github.io/amelia-dataset/) to download the dataset.

Once downloaded, create a symbolic link into  `datasets`:

```bash
cd datasets
ln -s /path/to/amelia .
```

### Installation

Make sure that you have [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) installed.

**Recommended:** Use the  [`install.sh`](https://github.com/AmeliaCMU/AmeliaScenes/blob/main/install.sh) to download and install the Amelia Framework:

```bash
chmod +x install.sh
./install.sh amelia
```

This will create a conda environment named `amelia` and install all dependencies.

Alternatively, refer to [`INSTALL.md`](https://github.com/AmeliaCMU/AmeliaScenes/blob/main/INSTALL.md) for manual installation.

**Note:** AmeliaTF requires the Amelia dataset and AmeliaScenes' dependencies to run, refer to AmeliaScenes' and AmeliaTF's installation.

#### Scenario Pre-processing

Once you've downloaded the dataset and installed the required modules. You need to post-process the dataset. To do so, follow the instructions [here](https://github.com/AmeliaCMU/AmeliaScenes/blob/main/README.md).

#### Additional Notes

Our repository's structure is based on this [template](https://github.com/ashleve/lightning-hydra-template), which uses Hydra and Pytorch Lightning. We recommend going through their [README](https://github.com/ashleve/lightning-hydra-template?tab=readme-ov-file#your-superpowers) for further details into the code's functionalities.

## How to use

Activate your amelia environment (**Please make sure to follow the pre-requisites guidelines above above**):

```bash
conda activate amelia
```

### Training a Model

The general format for running a training experiment is:

```bash
python src/train.py data=<data_config> model=<model_config> trainer=<trainer_config>
```

where:

- `<data_config>`, represents a dataset configuration specified under `./configs/data`
- `<model_config>`, represents a model configuration specified under `./configs/model`
- `<trainer_config>`, represents the trainer to be used, (e.g., CPU, GPU, DDP, etc), specified under `./configs/trainer`

For example, to train our model on GPU using all of our currently supported airports, you would run:

```bash
python src/train.py data=seen-all model=marginal trainer=gpu
```

### Evaluating a Model

If you already have a pre-trained checkpoint you can run evaluation only using `eval.py` and following a similar format as above. However, you need to provide the path to the pre-trained weights. For example,

```bash
python src/eval.py data=seen-all model=marginal trainer=gpu ckpt_path=/path/to/pretrained/weights.ckpt
```

### Our experiments

We provide the configuration combination to run our experiments, as well as our pre-trained weights.

#### Single-Airport Experiments (Table 5 in our paper)

The model configuration used for all of these experiments was `marginal.yaml`.

| Airport                                   | Airport ICAO | Data Config | ADE@20 | FDE@20 | ADE@50 | FDE@50 | Weights  |
|:-----------------------------------------:|:------------:|:-----------:|:------:| :----: | :----: | :----: | :------: |
| Ted Stevens Anchorage Intl. Airport       |      PANC    | `panc.yaml` | 10.11  | 20.87  | 38.84  | 101.89 | [panc](https://airlab-share-01.andrew.cmu.edu:9000/amelia-processed/Single-Airport/panc.zip) |
| Boston-Logan Intl. Airport                |      KBOS    | `kbos.yaml` |  5.58  | 10.90  | 21.34  |  53.76 | [kbos](https://airlab-share-01.andrew.cmu.edu:9000/amelia-processed/Single-Airport/kbos.zip) |
| Ronald Reagan Washington Natl. Airport    |      KDCA    | `kdca.yaml` |  4.74  |  9.22  | 16.42  |  40.57 | [kdca](https://airlab-share-01.andrew.cmu.edu:9000/amelia-processed/Single-Airport/kdca.zip) |
| Newark Liberty Intl. Airport              |      KEWR    | `kewr.yaml` |  6.61  | 12.92  | 23.68  |  57.63 | [kewr](https://airlab-share-01.andrew.cmu.edu:9000/amelia-processed/Single-Airport/kewr.zip) |
| John F. Kennedy Intl. Airport             |      KJFK    | `kjfk.yaml` |  4.58  |  9.52  | 17.11  |  41.19 | [kjfk](https://airlab-share-01.andrew.cmu.edu:9000/amelia-processed/Single-Airport/kjfk.zip) |
| Los Angeles Intl. Airport                 |      KLAX    | `klax.yaml` | 11.36  | 20.63  | 36.08  |  88.25 | [klax](https://airlab-share-01.andrew.cmu.edu:9000/amelia-processed/Single-Airport/klax.zip) |
| Chicago-Midway Intl. Airport              |      KMDW    | `kmdw.yaml` |  3.30  |  6.12  | 11.50  |  28.80 | [kmdw](https://airlab-share-01.andrew.cmu.edu:9000/amelia-processed/Single-Airport/kmdw.zip) |
| Louis Armstrong New Orleans Intl. Airport |      KMSY    | `kmsy.yaml` |  2.73  |  5.12  |  9.89  |  25.68 | [kmsy](https://airlab-share-01.andrew.cmu.edu:9000/amelia-processed/Single-Airport/kmsy.zip) |
| Seattle-Tacoma Intl. Airport              |      KSEA    | `ksea.yaml` |  9.76  | 18.35  | 29.94  |  65.82 | [ksea](https://airlab-share-01.andrew.cmu.edu:9000/amelia-processed/Single-Airport/ksea.zip) |
| San Francisco Intl. Airport               |      KSFO    | `ksfo.yaml` |  5.06  |  9.82  | 17.05  |  40.23 | [ksfo](https://airlab-share-01.andrew.cmu.edu:9000/amelia-processed/Single-Airport/kssfo.zip) |

<hr>

#### Multi-Airport Experiments (Table 6 and 8 in our paper)

The model configuration used for all of these experiments was also `marginal.yaml`.

| Seen Airport(s)                                            | Unseen Airport(s)                                    | Data Config     | Avg. ADE@20 | Avg. FDE@20 | Avg. ADE@50 | Avg. FDE@50 | Weights |
| :--------------------------------------------------------: | :--------------------------------------------------: | :-------------: | :---------: | :---------: | :---------: | :---------: | :-----: |
| KMDW                                                       | KEWR, KBOS, KSFO, KSEA, KDCA, PANC, KLAX, KJFK, KMSY | `seen-1.yaml`   |     3.30    |     6.12    |    11.50    |    28.80    | [seen-1](https://airlab-share-01.andrew.cmu.edu:9000/amelia-processed/Muti-Airport/seen-1.zip) |
| KMDW, KEWR                                                 | KBOS, KSFO, KSEA, KDCA, PANC, KLAX, KJFK, KMSY       | `seen-2.yaml`   |     3.31    |     6.23    |    11.84    |    28.89    | [seen-2](https://airlab-share-01.andrew.cmu.edu:9000/amelia-processed/Muti-Airport/seen-2.zip) |
| KMDW, KEWR, KBOS                                           | KSFO, KSEA, KDCA, PANC, KLAX, KJFK, KMSY             | `seen-3.yaml`   |     3.26    |     6.59    |    12.46    |    31.81    | [seen-3](https://airlab-share-01.andrew.cmu.edu:9000/amelia-processed/Muti-Airport/seen-3.zip) |
| KMDW, KEWR, KBOS, KSFO                                     | KSEA, KDCA, PANC, KLAX, KJFK, KMSY                   | `seen-4.yaml`   |     3.52    |     6.74    |    12.71    |    31.64    | [seen-4](https://airlab-share-01.andrew.cmu.edu:9000/amelia-processed/Muti-Airport/seen-4.zip) |
| KMDW, KEWR, KBOS, KSFO, KSEA, KDCA, PANC                   | KLAX, KJFK, KMSY                                     | `seen-7.yaml`   |     3.59    |     7.03    |    14.35    |    38.62    | [seen-7](https://airlab-share-01.andrew.cmu.edu:9000/amelia-processed/Muti-Airport/seen-7.zip) |
| KMDW, KEWR, KBOS, KSFO, KSEA, KDCA, PANC, KLAX, KJFK, KMSY | -                                                    | `seen-all.yaml` |     3.88    |     7.70    |    15.30    |    40.91    | [seen-all](https://airlab-share-01.andrew.cmu.edu:9000/amelia-processed/Muti-Airport/seen-all.zip) |

<hr>

#### Other Experiments

- We trained our models under a **marginal** prediction setting, but we have support for training models on a **joint** prediction setting. To change the prediction paradigm, change the model parameter to `joint`. For example:

```bash
python src/train.py data=seen-all model=joint trainer=gpu
```

- Our model can be trained with and without context (maps). To train the trajectory-only model, use either `marginal_traj` or `joint_traj` configurations. For example,

```bash
python src/train.py data=seen-all model=marginal_traj trainer=gpu
```

<hr>

## BibTeX

If you find our work useful in your research, please cite us!

```bibtex
@inbook{navarro2024amelia,
  author = {Ingrid Navarro and Pablo Ortega and Jay Patrikar and Haichuan Wang and Zelin Ye and Jong Hoon Park and Jean Oh and Sebastian Scherer},
  title = {AmeliaTF: A Large Model and Dataset for Airport Surface Movement Forecasting},
  booktitle = {AIAA AVIATION FORUM AND ASCEND 2024},
  chapter = {},
  pages = {},
  doi = {10.2514/6.2024-4251},
  URL = {https://arc.aiaa.org/doi/abs/10.2514/6.2024-4251},
  eprint = {https://arc.aiaa.org/doi/pdf/10.2514/6.2024-4251},
}
```
