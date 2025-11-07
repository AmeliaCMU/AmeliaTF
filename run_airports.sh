python3 amelia_tf/train.py data=kmdw model=marginal trainer=gpu trainer.devices=[1] data.dataset.config.sampling_strategy=critical logger.wandb.project=amelia_fix_rot model.optimizer.lr=0.0001 data.dataset.config.min_agents=2

python3 amelia_tf/train.py data=kewr model=marginal trainer=gpu trainer.devices=[1] data.dataset.config.sampling_strategy=critical logger.wandb.project=amelia_fix_rot model.optimizer.lr=0.0001 data.dataset.config.min_agents=2

python3 amelia_tf/train.py data=kbos model=marginal trainer=gpu trainer.devices=[1] data.dataset.config.sampling_strategy=critical logger.wandb.project=amelia_fix_rot model.optimizer.lr=0.0001 data.dataset.config.min_agents=3

python3 amelia_tf/train.py data=ksfo model=marginal trainer=gpu trainer.devices=[1] data.dataset.config.sampling_strategy=critical logger.wandb.project=amelia_fix_rot model.optimizer.lr=0.00001 data.dataset.config.min_agents=2

python3 amelia_tf/train.py data=ksea model=marginal trainer=gpu trainer.devices=[1] data.dataset.config.sampling_strategy=critical logger.wandb.project=amelia_fix_rot model.optimizer.lr=0.0001 data.dataset.config.min_agents=2

python3 amelia_tf/train.py data=kdca model=marginal trainer=gpu trainer.devices=[1] data.dataset.config.sampling_strategy=critical logger.wandb.project=amelia_fix_rot model.optimizer.lr=0.00001 data.dataset.config.min_agents=2

python3 amelia_tf/train.py data=panc model=marginal trainer=gpu trainer.devices=[1] data.dataset.config.sampling_strategy=critical logger.wandb.project=amelia_fix_rot model.optimizer.lr=0.00001 data.dataset.config.min_agents=2

python3 amelia_tf/train.py data=klax model=marginal trainer=gpu trainer.devices=[1] data.dataset.config.sampling_strategy=critical logger.wandb.project=amelia_fix_rot model.optimizer.lr=0.00001 data.dataset.config.min_agents=2

python3 amelia_tf/train.py data=kmsy model=marginal trainer=gpu trainer.devices=[1] data.dataset.config.sampling_strategy=critical logger.wandb.project=amelia_fix_rot model.optimizer.lr=0.0001 data.dataset.config.min_agents=3

python3 amelia_tf/train.py data=kjfk model=marginal trainer=gpu trainer.devices=[1] data.dataset.config.sampling_strategy=critical logger.wandb.project=amelia_fix_rot model.optimizer.lr=0.0001 data.dataset.config.min_agents=2
