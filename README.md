![LOGO](https://github.com/DeepWave-Kaust/SSL-Resolution-Enhancement/blob/main/asset/logo.jpg)

Reproducible material for **Self-supervised Seismic Resolution Enhancement - Shijun Cheng, Haoran Zhang, and Tariq Alkhalifah.**

# Project structure
This repository is organized as follows:

* :open_file_folder: **ssl_resolution**: python library containing routines for self-supervised seismic resolution enhancement;
* :open_file_folder: **asset**: folder containing logo;
* :open_file_folder: **data**: folder to store dataset;
* :open_file_folder: **scripts**: set of shell scripts used to run training and testing

## Supplementary files
To ensure reproducibility, we provide the the data set for training and testing, and the trained model for validate the low-frequency extrapolation performance. 

* **Training and Testing data set**
Download the training and testing data set [here](https://drive.google.com/file/d/17QtAFV4HBDtq2dc6bOY91yvGtt3fDbsk/view?usp=sharing). Then, use `unzip` to extract the contents to `dataset/train` and `dataset/test`.

* **Trained model:**
Download our trained model [here](https://drive.google.com/file/d/1tJd4N7XBtmM2UzceSJegHCAxxlnRQQyw/view?usp=sharing). Then, extract the contents.

## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. Activate the environment by typing:
```
conda activate  ssl_resolution
```

After that you can simply install your package:
```
pip install .
```
or in developer mode:
```
pip install -e .
```

## Configuration file for training and testing:
The training and testing configuration is organized in the config file `config.yaml`. You can modify the configuration value according to your situation.

## Scripts :page_facing_up:
When you have downloaded the supplementary files and have installed the environment, you can entry the scripts file folder and run demo. We provide two scripts which are responsible for training and testing examples.

Under our framework, for training, you need to set configuration value of args (`train_mode`) in config file `config.yaml` as `SSL`, and then directly run:
```
sh run_train.sh
```

If you need to compare with a framework only use warm-up phase, you need to set configuration value of args (`train_mode`) in config file `config.yaml` as `SL`, and then run:
```
sh run_train.sh
```

For testing, you can directly run:
```
sh run_test.sh
```

**Note:** We emphasize that the training logs is saved in the `runs/` file folder. You can use the `tensorboard --logdir=./` or extract the log to view the changes of the metrics as a function of epoch.

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce A100 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU. Due to the high memory consumption during the training phase, if your graphics card does not support large batch training, please reduce the configuration value of args (`batch_size`) in config file `config.yaml`.

## Cite us
```bibtex
@ARTICLE{cheng2024sslresolution,
  author={Cheng, Shijun and Zhang, Haoran and Alkhalifah, Tariq},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Self-supervised Seismic Resolution Enhancement}, 
  year={2025},
  volume={63},
  number={},
  pages={1-15},
  doi={10.1109/TGRS.2025.3528414}}


