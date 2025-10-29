python3 amelia_tf/train.py data=kbos model=marginal trainer=gpu trainer.devices=[1] data.dataset.config.sampling_strategy=critical logger.wandb.project=amelia_fix_rot model.optimizer.lr=0.0001 data.dataset.config.min_agents=3

python3 amelia_tf/train.py data=kdca model=marginal trainer=gpu trainer.devices=[1] data.dataset.config.sampling_strategy=critical logger.wandb.project=amelia_fix_rot model.optimizer.lr=0.00001 data.dataset.config.min_agents=2

python3 amelia_tf/train.py data=ksfo model=marginal trainer=gpu trainer.devices=[1] data.dataset.config.sampling_strategy=critical logger.wandb.project=amelia_fix_rot model.optimizer.lr=0.0001 data.dataset.config.min_agents=2