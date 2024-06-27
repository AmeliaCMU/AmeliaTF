# Experiments (KSEA)

| Experiment Config | Data Config | Marginal | Joint | Agent Type | Context |  
| :--------: | :---: | :---: | :---: | :---: | :---: | 
| ```exp1.yaml``` | ```swim_ksea_traj.yaml``` | &check; |  |  |  |
| ```exp2.yaml``` | ```swim_ksea_traj.yaml``` | | &check; |  | |
| ```exp3.yaml``` | ```swim_ksea_atype.yaml``` | &check; |   | &check; |  |
| ```exp4.yaml``` | ```swim_ksea_atype.yaml``` |  | &check;  | &check; |  |
| ```exp5.yaml``` | ```swim_ksea_map.yaml``` | &check; |  |  | v0 |
| ```exp6.yaml``` | ```swim_ksea_map.yaml``` |  | &check; |  | v0 |
| ```exp7.yaml``` | ```swim_ksea_map.yaml``` | &check; |  |  | v1 |
| ```exp8.yaml``` | ```swim_ksea_map.yaml``` |  | &check; |  | v1 |
| ```exp9.yaml``` | ```swim_ksea_map.yaml``` | &check; |  |  | v2 |
| ```exp10.yaml``` | ```swim_ksea_map.yaml``` |  | &check; |  | v2 |
| ```exp11.yaml``` | ```swim_ksea_map.yaml``` |  | &check; |  | v3 |
| ```exp12.yaml``` | ```swim_ksea_map.yaml``` |  | &check; |  | v3 |


To run an experiment:
```
python src/train.py data=[data_config] model=[model_config]
```

Example for experiment 1:
```
python src/train.py data=swim model=exp1
```

# Implementation Details

For running distributed training on systems with 4090 it may be possible that the NCCL backend runs into a deadlock and causes the code to hang. To prevent this, set the following environment variable before training.

```
export NCCL_P2P_DISABLE=1
```