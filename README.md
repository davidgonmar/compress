# 🤏 compress: A PyTorch toolkit for model compression

Lightweight utilities for compressing PyTorch models via quantization, sparsity/pruning, low‑rank factorization, layer fusion, and knowledge distillation.

The library includes utilities for some compression techniques. In general, these have been mainly tested on smaller models, but should work on larger models as well.

In general, the objective of the library is not to be neither a performance-optimized nor a production ready solution, it is rather a research and prototyping toolkit that I use to experiment with.

## Requirements
The project has been developed and tested on Python 3.11
Install Python dependencies listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

Install the package (editable for development):

```bash
pip install -e .
```

Alternatively, you can install it with
```bash
pip install git+https://github.com/davidgonmar/compress
```

## Information

The library is organized in different modules. As I use the code for my own exploration, it would be difficult to maintain a precise documentation of the functions and functionalities, but a general overview is provided below as it is useful.

In general, both nn.Conv2d and nn.Linear layers are supported, and the library is designed to be extensible to other layer types.

### Quantization

Implements **post-training quantization (PTQ)** techniques with both online and offline quantization. Different calibration techniques are offered. Both symmetric and asymmetric quantization are supported (depending on the specific technique).

In general, when doing offline quantization, a batch of data needs to be passed to calibrate the quantization parameters.

Regular **quantization-aware training (QAT)** is also supported.  At the moment, weights are quantized at each iteration, while inputs use an observer. The specific calibration technique can be chosen. Moreover, **learnable quantization** is supported through **LSQ** (basically computing the "natural" derivatives of the quantization function by using a straight-through estimator). Here, both weights and inputs are initially calibrated with a technique of choice, and then the quantization parameters are learned during training.

### Sparsity / Pruning

Implements different **structured pruning** and **unstructured pruning** techniques. The library includes both **magnitude-based pruning** and **gradient-based pruning** (e.g., using the Taylor expansion of the loss function). For details, see the code.

The library supporst sparsifying a model, so things like iterative pruning need to be handled externally. However, it supports a set of **schedulers** to schedule sparsity ratios, and pruned models support re-pruning (and fine-tuning) in sensible ways.

### Low-rank factorization

Implements different **low-rank factorization** techniques for compressing linear and convolutional layers. The library includes both classic **SVD factorization** and more advanced **activation-aware factorization** techniques.

It allows the user to manually select the rank of the factorization for each layer following different criteria (e.g., energy-based, rank-based, etc.). It also includes an experimental **automatic rank selection** method based on a global energy criterion.

It also includes some utilities for **low-rank** regularization.

### More

The library also implements useful utilities for things like **layer fusion** (e.g., fusing batchnorm layers into convolutional layers, this is useful to mimic inference behavior of some CNNs), **FLOPs and parameter counting**, and **knowledge distillation** (e.g., implementing different losses for distillation). It also includes experiment utilities (e.g., for evaluation), including implementation of some models that are not available in popular libraries (e.g. CIFAR ResNets).

## Remarks

This is a personal project that I use to experiment with model compression techniques. Some compression techniques are heavily inspired by existing works but have some differences (for research purposes or practical implementation reasons, for instance), while others (including removed ones present in the git history) are my own explorations.


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.