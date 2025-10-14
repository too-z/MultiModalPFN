# Multi-Modal PFN (Prior-data Fitted Network)

## ✨ Introduction

**MMPFN** is an extension of **TabPFN**, designed to handle **multimodal data** — combining tabular, image, and text inputs in a unified learning framework. While TabPFN has shown strong performance on purely tabular datasets, it lacks the ability to integrate heterogeneous modalities.

Comprehensive experiments on datasets show that MMPFN **outperforms state-of-the-art baselines**, efficiently leveraging diverse data types to enhance predictive performance. This demonstrates the potential of extending **prior-data fitted networks** into the multimodal domain, offering a scalable and effective solution for heterogeneous data learning.

## 🚦 Getting started
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

## 📜 License
This project follows the original TabPFN license policy.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0



Copy tsp/datasets/openml directory into openml cache (default: ~/.cache/)
