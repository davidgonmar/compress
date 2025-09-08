## compress: A PyTorch toolkit for model compression

Lightweight utilities for compressing PyTorch models via quantization, sparsity/pruning, low‑rank factorization, layer fusion, and knowledge distillation.

The library includes utilities for some compression techniques. In general, these have been mainly tested on smaller models, but should work on larger models as well.

In general, the objective of the library is not to be neither a performance-optimized nor a production ready solution, it is rather a research and prototyping toolkit that I use to experiment with.

### Requirements
The project has been developed and tested on Python 3.11
Install Python dependencies listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

Install the package (editable for development):

```bash
pip install -e .
```

Otherwise, you can install it with
```bash
pip install git+https://github.com/davidgonmar/compress
```