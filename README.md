# mech439_pybullet
Pybullet simulation framework for a MECH439 lecture

___
## Environments
* Windows 10/11, Ubuntu 20.04/22.04
* Python 3.10
___
## Installation Guide
1. Create conda environment.
```shell
$ conda create -n <ENV_NAME> python=3.10    # create virtual environment
$ conda activate <ENV_NAME>
$ pip install -r requirements.txt           # install dependencies
$ conda install pinocchio==2.7.0 -c conda-forge    # install pinocchio (Rigid Body Dynamics Library)
```
---
## Build a docs
1. Build a sphinx docs.
```shell
$ cd ./docs
$ make html
```
2. Open built html file located in <span style='color: #2D3748; background-color: #f6f8fa'>./docs/html/index.html</span>.
