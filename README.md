# DL4S2S: Deep Learning for Seasonal-to-Seasonal Forecasting

![Version](https://img.shields.io/badge/version-0.0.1-green)

## Table of Contents
1. [Overview](#overview)
2. [Repository Contents](#library)
3. [Preprocessing Instructions](#data)
   
   3.1.[Data](#download)
   
   3.2.[Data Preprocessing](#preprocesing)
   
4. [Training Instructions](#Experiments)
   
    4.1 [Hyperparameter Tuning](#training)
   
    4.2 [Ensemble Training](#Baseline)
   
    4.3 [Performance analysis](#Network)
   
5. [Result Instruction](#resutls)

	5.1 [NAE precursors](#NAE)
   
    4.2 [SPV Index](#SPV)
   
    4.3 [MJO Phases](#MJO)

6. [Plots](#Plots)

7. [Further references](#Refs)

## Overview


This repository contains the code and supplementary packages for the paper **["Deep Learning Meets Teleconnections: Improving S2S Predictions for European Winter Weather"]()**  by Bommer et. al.



#### Please note that while the code basis is complete, full reproducability updates will be made as soon as the peer review is finished


**Motivation**

DL4S2S is a Python package for developing and applying deep learning models to subseasonal-to-seasonal forecasting tasks. The package provides a framework for building, training, and evaluating various deep learning architectures, including an Long Short-term Memory (LSTM) network, a Vision-Transformer 

</p>
<p align="center">
  <img width="600" src="https://github.com/philine-bommer/DL4S2S/blob/main/FinalFigureDL4S2S.png">


**Citation**

Please cite:
```bibtex
@article {
}
```

## Directory Structure
-----------------------

The DL4S2S directory is organized into the following directories:

* `deepS2S`: Package that contains the core deep learning models, datasets, and utilities.
	+ `model`: Defines the deep learning architectures, including ViT, LSTM, and CNN.
	+ `dataset`: Provides datasets for seasonal-to-seasonal forecasting, including data loading and preprocessing.
	+ `utils`: Offers various utility functions for data manipulation, visualization, and evaluation.
* `Experiments`: Holds experiment-specific code, including training scripts and configuration files.
	+ `Training`: Contains training scripts for different models and datasets.
	+ `Appendix`: Provides additional scripts and utilities for data analysis and visualization.

## Installation
---------------

To install the DL4S2S package, run the following command:
```bash
cd DeepS2S
pip install -e .
```
This will install the **deepS2S** package and its dependencies.

## Usage
-----

To use the DL4S2S package, follow these steps:

1. Import the necessary modules: `from deepS2S import *`
2. Load a dataset using the `dataset` module: `data = TransferData(...)`
3. Build a model using the `model` module: `model = ViTLSTM(...)`
4. Train the model using the `Training` scripts: `python train.py ...`
5. Evaluate the model using the `utils` module: `evaluate_accuracy(model, data)`

**Contributing**
-----------------

Contributions to the DL4S2S package are welcome. If you'd like to contribute, please fork the repository, make your changes, and submit a pull request.

**License**
-----------

The DL4S2S package is licensed under the MIT License.

**Acknowledgments**
----------------

The DL4S2S package was developed by Philine L. Bommer and is based on the work of Paul Boehnke.