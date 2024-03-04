# perception-guarantees
Code for combining generalization guarantees for perception and planning.

## Installation

Install [`iGibson`](https://stanfordvl.github.io/iGibson/installation.html). Also download assets and datasets. 

Install [`Meshlab`](https://www.meshlab.net/) for visualizing point clouds and debugging.

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

## Generating the calibration dataset with Pybullet sim (in nav_sim)

After following the installation instructions in the nav_sim README, run script to generate the room configurations (change folder name as required):
```console
python nav_sim/asset/get_room_from_3dfront.py --save_task_folder=<path to save dataset>/rooms --mesh_folder=<path to save dataset>/3D-FUTURE-model-tiny
```

Generate task dataset by aggregating the room configurations:
```console
python nav_sim/asset/get_task_dataset.py --save_path=<path to save dataset>/task.pkl --task_folder=<path to save dataset>/rooms
```

Collect a calibration dataset with random tasks generated:
```console
python nav_sim/test/test_task_sim.py --task_dataset=<path to save dataset>/task.pkl --save_dataset=<path to save dataset>/
```
Use above task dataset to test the environment with random locations in each room. This code will 
1. Generate the pointclouds through the pybullet sim (we're using the ZED2i camera parameters)
2. Compute the features using 3DETR for each pointcloud in each location of every environment
3. This will generate the following files: `data/features.pt`, `data/bbox_labels.pt`, `data/loss_mask.pt`, and `data/finetune.pt`. This is the calibration dataset.

Sim datset generation:
Repeat the first two steps and save a new task_sim.pkl file with new rooms:
```console
python nav_sim/asset/get_room_from_3dfront.py --save_task_folder=<path to save dataset>/rooms_sim --mesh_folder=<path to save dataset>/3D-FUTURE-model-tiny --num_room=100 --seed=33 --sim=True

Generate task dataset by aggregating the room configurations:
```console
python nav_sim/asset/get_task_dataset.py --save_path=<path to save dataset>/task_sim.pkl --task_folder=<path to save dataset>/rooms_sim
```

## Obtain the CP inflation bound uding the calibration dataset
Run the code to get the CP inflation bound:
```commandline
python cp_bound.py
```
If you want to finetune the outputs from 3DETR (using a split CP) and then use this to get a CP bound (using "dataset"):
```commandline
python cp_bound_with_finetuning.py
```

## Obtain the CP inflation bound uding the calibration dataset

## Notes

- Be careful with reference frames; here are the main ones:
  - Coordinate system of scene:  +X (right), +Y (forward), +Z (up)
  - Coordinate system of camera: +X (right), +Y (up), +Z (backwards)
  - Coordinate system for 3DETR: same as scene


