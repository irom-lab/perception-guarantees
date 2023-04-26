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
python nav_sim/test/test_task_sim.py
```

Test vectorized environment:

```console
python nav_sim/test/test_vec_sim.py
```

### Room generation

1. Download the 5.1G furniture mesh data from the 3D-Front dataset (Allen has the whole original dataset, but it takes a while to process them). Right now, using 397 chair meshes.
```console
pip install gdown
gdown 1VaKNDAq9iQvxtvDLGLmA6k7Hl0L0AJ7W
pigz -dc 3D-FUTURE-model-tiny.file | pv | tar xf -           # fast unzip
```

2. Run script to generate the room configurations:
```console
python nav_sim/asset/get_room_from_3dfront.py --save_task_folder=[folder to save the room configurations] --mesh_folder=[3D-Front model folder] --num_room --num_room_per_furniture --room_dim --min_obstacle_spacing --min_init_goal_dist
```
3. Generate task dataset by aggregating the room configurations:
```console
python nav_sim/asset/get_task_dataset.py --save_path=[path to save the task dataset] --task_folder=[folder to the saved room configurations]
```
