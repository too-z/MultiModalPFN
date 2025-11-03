# Multi-Modal PFN (Prior-data Fitted Network)

![Crates.io](https://img.shields.io/crates/l/Ap?color=orange)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)

## Introduction 
> **MMPFN** is an extension of **TabPFN**, designed to handle **multimodal data** — combining tabular, image, and text inputs in a unified learning framework. While TabPFN has shown strong performance on purely tabular datasets, it lacks the ability to integrate heterogeneous modalities.
>
> Comprehensive experiments on datasets show that MMPFN **outperforms state-of-the-art baselines**, efficiently leveraging diverse data types to enhance predictive performance. This demonstrates the potential of extending **prior-data fitted networks** into the multimodal domain, offering a scalable and effective solution for heterogeneous data learning.

## Set-up

Conda Environment
```
conda env create -f environment.yaml
```

Install
```
python setup.py develop
```

Place the checkpoint file and dataset in their respective locations, then update the model_path as shown below:

```
ln -s /path/to/model/params # symlink parameter
ln -s /path/to/data # symlink data

model_path = Path(__file__).parent/ "parameters" / "tabpfn-v2-classifier.ckpt"
```

## Usage


To reproduce the experimental results, you can run `run_pad_ufes_20_mmpfn.py`, which uses Optuna to explore all hyperparameters.  
```
python run_pad_ufes_20_mmpfn.py
```

To view the results obtained with the optimized parameters, open and execute the notebook file `run_pad_ufes_20_mmpfn.ipynb`.


## 📘 License
This project follows the original TabPFN license policy(Apache 2.0 with additional attribution requirement): [here](https://priorlabs.ai/tabpfn-license/)
