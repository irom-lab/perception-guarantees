# perception-guarantees
Code for combining generalization guarantees for perception and planning.

## Installation

Install [`iGibson`](https://stanfordvl.github.io/iGibson/installation.html). Also download assets and datasets. 

Install CUDA 10.2.

Install PyTorch 1.9.0:
```
pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Install pointnet2:
```
cd third_party/pointnet2 && python setup.py install
```

Install plyfile:
```
pip install plyfile
```

Install Cython:
```
conda install cython
```

Compile cythonized version of loss function:
```
cd utils && python -E cython_compile.py build_ext --inplace
```

