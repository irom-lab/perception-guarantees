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

Test single environment with assets in `nav_sim/asset`:

```console
python nav_sim/test/test_vanilla_sim.py
```

Test environment with random tasks generated after following the instructions in the next section:

```console
python nav_sim/test/test_task_sim.py --task_dataset=/home/anushri/Documents/Projects/data/perception-guarantees/task.pkl --save_dataset=/home/anushri/Documents/Projects/data/perception-guarantees/task.npz
```

Test vectorized environment:

```console
python nav_sim/test/test_vec_sim.py
```

### Room generation

1. Get the 3D-Front model dataset (TODO: upload to drive). Right now, using 397 chair meshes.
2. Run script to generate the room configurations:
```console
python nav_sim/asset/get_room_from_3dfront.py --save_task_folder=/home/anushri/Documents/Projects/data/perception-guarantees/rooms --mesh_folder=/home/anushri/Documents/Projects/data/perception-guarantees/3D-FUTURE-model-tiny
```
3. Generate task dataset by aggregating the room configurations:
```console
python nav_sim/asset/get_task_dataset.py --save_path=/home/anushri/Documents/Projects/data/perception-guarantees/task.pkl --task_folder=/home/anushri/Documents/Projects/data/perception-guarantees/rooms
```
