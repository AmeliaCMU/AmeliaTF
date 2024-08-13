from setuptools import find_packages, setup

setup(
    name="amelia_tf",
    version="1.0.0",
    description="Amelia TF is a large transformer-based trjectory forecasting model",
    author="",
    author_email="",
    packages=find_packages(['./src/*']),
    url="https://github.com/AmeliaCMU/AmeliaTF",
    install_requires=[  # missing amelia_scenes
        'setuptools',
        'lightning==2.2.0.post0',
        'hydra-core==1.3.2',
        'hydra_colorlog==1.2.0',
        'torch==2.4.0',
        'pytorch-lightning==2.2.0.post0',
        'wandb==0.15.11',
        'pyrootutils==1.0.4',
        'rich==13.7.1',
        'opencv-python==4.7.0.72',
        'numpy==1.21.2',
        'imageio==2.34.0',
        'easydict==1.10',
        'matplotlib==3.7.1',
        'scipy==1.9.1',
        'geographiclib==2.0',
        'pandas==2.0.3',
        'tqdm==4.65.0',
        'networkx==3.1',
        'python-dateutil==2.9.0',


    ],
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
