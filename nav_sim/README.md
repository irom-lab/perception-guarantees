## Running navigation simulation in PyBullet

Tested with Ubuntu 20.04.

### Installation

Navigate to the root directory and install the conda environment:

```console
conda env create -f conda_ubuntu20.yml
```

Install this repository locally:

```console
pip install -e .
```

### Usage

Test single environment:

```console
python nav_sim/test/test_sim.py
```

Test vectorized environment:

```console
python nav_sim/test/test_vec_sim.py
```
