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

## (OLD) Running the code with iGibson sim

Generate point clouds from different locations in different environments:
```commandline
python render_env_parallel.py
```
You can set the number of environments in main(); other parameters (e.g., range of camera locations) 
are in render_env(). This generates a file `data/training_data_raw.npz`.

Next, compute features for each point cloud in each environment using 3DETR:
```commandline
python compute_features.py
```
This will generate the following files: `data/features.pt`, `data/bbox_labels.pt`, and `data/loss_mask.pt`.

Run the training loop:
```commandline
python train.py
```

The neural network model is in `model_perception.py`. Currently, it is a simple MLP. The training code will 
generate a model which is saved in `trained_models/perception_model`.

Finally, one can visualize the predictions of the model and compare these with the ground truth bounding
boxes (and the bounding boxes generated by 3DETR) by running the following:
```commandline
python viz_model_prediction.py
```
This code will generate .ply files in the `viz_pcs` folder. You can visualize them using [`Meshlab`](https://www.meshlab.net/).

## (NEW) Running the code with Pybullet sim (in nav_sim)

After following the installation instructions in the nav_sim README, run script to generate the room configurations (change folder name as required):
```console
python nav_sim/asset/get_room_from_3dfront.py --save_task_folder=/home/anushri/Documents/Projects/data/perception-guarantees/rooms --mesh_folder=/home/anushri/Documents/Projects/data/perception-guarantees/3D-FUTURE-model-tiny
```

Generate task dataset by aggregating the room configurations:
```console
python nav_sim/asset/get_task_dataset.py --save_path=/home/anushri/Documents/Projects/data/perception-guarantees/task.pkl --task_folder=/home/anushri/Documents/Projects/data/perception-guarantees/rooms
```

Test environment with random tasks generated:
```console
python nav_sim/test/test_task_sim.py --task_dataset=/home/anushri/Documents/Projects/data/perception-guarantees/task.pkl
```
Use above task dataset to test the environment with random locations in each room. This code will 
1. Generate the pointclouds through the pybullet sim (we're using the ZED2i camera parameters)
2. Compute the features using 3DETR for each pointcloud in each location of every environment
3. This will generate the following files: `data/features.pt`, `data/bbox_labels.pt`, and `data/loss_mask.pt`

Run the training loop:
```commandline
python train.py
```

If this doesn't work, try:
```commandline
PYTORCH_JIT=0 python train.py
```

The above code will finetune the outputs from 3DETR (using the "prior dataset") and then use this to get a CP bound (using "dataset"). The code that is commented out provides a similar guarantee using PAC-Bayes. Then we evaluate the bounds from CP and Pac-Bayes and compare the two (using "test dataset").

Note that compute_features_nav_sim.py is no longer needed and has been combined into test_task_sim.py

## Notes

- Be careful with reference frames; here are the main ones:
  - Coordinate system of scene:  +X (right), +Y (forward), +Z (up)
  - Coordinate system of camera: +X (right), +Y (up), +Z (backwards)
  - Coordinate system for 3DETR: same as scene
- Need to handle masking of loss in case where object is not visible in a proper way.
   - Currently, we're just looking at the 3DETR probability in
    compute_features.py to say if the object is visible or not. But, it might be the case that the object is actually
    visible, but this does not cach it. We should mask the loss with 0 if 3DETR says that there is no object in the scene
    and if there is actually no object in the scene. See InFOVOfAgent function.
   - Need to make sure this is consistent when using the model elsewhere (e.g., evaluation).
- If I need to speed up training, I can use features just from the query that corresponds to an object.
- Code currently assumes that there is only one object in environment
      - render_env_parallel.py will need to output multiple bounding boxes for multiple objects.
      - compute_features.py will need a small modification to save multiple bounding boxes.
      - model_perception.py will need to output multiple bounding boxes.
      - loss function will need to handle multiple bounding boxes.
- Bounding boxes that gibson provides are slightly larger than they need to be (not entirely sure why).


