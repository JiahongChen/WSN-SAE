# WSN Sampling Optimization using Spatiotemporal Autoencoder
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)

This is the Python+TensorFlow code to reproduce results for paper ['WSN Sampling Optimization for Signal
Reconstruction using Spatiotemporal Autoencoder'](https://ieeexplore.ieee.org/document/9133473).

<a href="url"><img src="GraphicAbstract.png" width="600" ></a>

Due to GitHub file size limitations, datasets are not upload to this repo, you can:
1. Download raw data from [NOAA](https://www.esrl.noaa.gov/psd/data/gridded/data.noaa.oisst.v2.html).
2. Request preprocessed data by sending email to me at jhchen@mech.ubc.ca.

# Requirements
* Platform : Linux 
* Computing Environment:
  * CUDA 10.1 
  * TensorFlow 1.14.0
* Packages: ```pandas, numpy, scipy, argparse```.
* Hardware (optional) : Nvidia GPU (SST requires around 7GB of GPU memory)

# Getting Started
1. Computing environment set up can be refered to [this repo](https://github.com/JiahongChen/Set-up-deep-learning-frameworks-with-GPU-on-Google-Cloud-Platform). 
1. Download data and place it at './Data' folder.
1. Run the code by
```
bash batchrun.sh
```

# Citation
Please cite our paper if you use our code for your work.
```
@article{chen2019optimization,
  author={J. {Chen} and T. {Li} and J. {Wang} and C. W. d. {Silva}},
  journal={IEEE Sensors Journal}, 
  title={WSN Sampling Optimization for Signal Reconstruction Using Spatiotemporal Autoencoder}, 
  year={2020},
  volume={20},
  number={23},
  pages={14290-14301},
  doi={10.1109/JSEN.2020.3007369}}
```

# To do
* Code for visualization
* Code for optimizing WSN sampling strategy
