# Perceive with Confidence
Code for the paper "Perceive With Confidence: Statistical Safety Assurances for Navigation with Learning-Based Perception".

Paper: https://arxiv.org/abs/2403.08185

Website: https://perceive-with-confidence.github.io/

## Installation

Download assets and datasets from 3D-Front: https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset

Replace the 'model_info.json' file contents with the 'model_info.json' file given in this code. The reason for this is we found that a lot of objects we used from the dataset were misclassified as chairs. The file provided here has removed the misclassified objects.

Run (for a cuda 11.7 install)
```
pip install -r requirements.txt
```
Install pointnet2:
```
cd third_party/pointnet2 && python setup.py install
```

## Run the planner code
To run a simple example: 1) download pre-computed samples [here](https://drive.google.com/drive/folders/1OJpOyWaD7Wnsqb16qsrWaPtZHYkquXiw?usp=sharing), 2) run `example_plan.py` (change the directory of downloaded samples accordingly). 
Alternatively, you can generate your own samples with `generate_samples.py`.

## Generating the calibration dataset with Pybullet sim (in nav_sim)

1. After following the installation instructions in the nav_sim README, run script to generate the room configurations (change folder name as required):
```console
python nav_sim/asset/get_room_from_3dfront.py --save_task_folder=<path to save dataset>/rooms --mesh_folder=<path to save dataset>/3D-FUTURE-model-tiny
```

2. Generate task dataset by aggregating the room configurations:
```console
python nav_sim/asset/get_task_dataset.py --save_path=<path to save dataset>/task.pkl --task_folder=<path to save dataset>/rooms
```

3. Collect a calibration dataset with random tasks generated:
```console
python nav_sim/test/test_task_sim.py --task_dataset=<path to save dataset>/task.pkl --save_dataset=<path to save dataset>/
```
Note that Line 68 in the above code uses the same samples that are eventually used in planning (assumed to be available in 'planning/pre_compute/Pset.pkl'). You can generate these samples using `generate_samples.py' as noted earlier.

Use above task dataset to test the environment with random locations in each room. This code will 
i) Generate the pointclouds through the pybullet sim (we're using the ZED2i camera parameters)
ii) Compute the features using 3DETR for each pointcloud in each location of every environment
iii) This will generate the following files: `data/features.pt`, `data/bbox_labels.pt`, `data/loss_mask.pt`, and `data/finetune.pt`. This is the calibration dataset.

4. Sim dataset generation:
Repeat the first two steps (1-2) and save a new task_sim.pkl file with new rooms:
```console
python nav_sim/asset/get_room_from_3dfront.py --save_task_folder=<path to save dataset>/rooms_sim --mesh_folder=<path to save dataset>/3D-FUTURE-model-tiny --num_room=100 --seed=33 --sim=True
```

5. Generate task dataset by aggregating the room configurations:
```console
python nav_sim/asset/get_task_dataset.py --save_path=<path to save dataset>/task_sim.pkl --task_folder=<path to save dataset>/rooms_sim
```

## Obtain the CP inflation bound using the calibration dataset
Run the code to get the CP inflation bound:
```commandline
python cp_bound.py
```
If you want to finetune the outputs from 3DETR (using split CP) and then use this to get a CP bound (using "dataset"):
```commandline
python cp_bound_with_finetuning.py
```

## Run experiments in sim
If you want to test the planner on many sim environments (generated the using steps 4-5 of the code used to generate the calibration dataset), run
```commandline
python planner_test_task.py
```
In this code the task folder is called rooms_sim, but you can change it to wherever you saved your test sim environment tasks. 
Note that if you have a finetuned model and you want to test that, set is_finetune = True

## Notes

- Be careful with reference frames; here are the main ones:
  - Coordinate system of scene:  +X (right), +Y (forward), +Z (up)
  - Coordinate system of camera: +X (right), +Y (up), +Z (backwards)
  - Coordinate system for 3DETR: same as scene


