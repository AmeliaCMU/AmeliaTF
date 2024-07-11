from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Describe Your Cool Project",
    author="",
    author_email="",
    url="https://github.com/user/project",
    install_requires=[  # missing amelia_viz
        "lightning",
        "hydra-core",
        "pyrootutils==1.0.4",
        "rich==13.7.1",
        "opencv-python==4.7.0.72,<4.8",
        'numpy==1.21.2,<2',
        'imageio==2.34.0,<3',
        'easydict==1.10',
        'matplotlib==3.7.1',
        'scipy==1.9.1',
        'pyproj==3.6.1',
        'geographiclib==2.0'
    ],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
