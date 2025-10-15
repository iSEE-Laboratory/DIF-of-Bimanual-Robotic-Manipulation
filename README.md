# Rethinking Bimanual Robotic Manipulation: Learning with Decoupled Interaction Framework
### (ICCV 2025) Official repository of paper "Rethinking Bimanual Robotic Manipulation: Learning with Decoupled Interaction Framework"

# Installation
## 0. Install Vulkan
```cmd
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools
```
## 1. Basic Env
First, prepare a conda environment:
```cmd
conda create -n RoboTwin python=3.8
conda activate RoboTwin
```
```cmd
pip install torch==2.4.1 torchvision sapien==3.0.0b1 scipy==1.10.1 mplib==0.1.1 gymnasium==0.29.1 trimesh==4.4.3 open3d==0.18.0 imageio==2.34.2 pydantic zarr openai huggingface_hub==0.25.0
```
Then, install pytorch3d:
```cmd
cd third_party/pytorch3d_simplified && pip install -e . && cd ../..
```
## 2. Modify mplib Library Code
### 2.1 Remove convex=True
You can use pip show mplib to find where the mplib installed.
```
# mplib.planner (mplib/planner.py) line 71
# remove `convex=True`

self.robot = ArticulatedModel(
            urdf,
            srdf,
            [0, 0, -9.81],
            user_link_names,
            user_joint_names,
            convex=True,
            verbose=False,
        )
=> 
self.robot = ArticulatedModel(
            urdf,
            srdf,
            [0, 0, -9.81],
            user_link_names,
            user_joint_names,
            # convex=True,
            verbose=False,
        )
```
### 2.2 Remove or collide
```
# mplib.planner (mplib/planner.py) line 848
# remove `or collide`

if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
                return {"status": "screw plan failed"}
=>
if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
                return {"status": "screw plan failed"}
```

# Usage
# 1. Task Running and Data Collection
Running the following command will first search for a random seed for the target collection quantity (default is 100), and then replay the seed to collect data.
```cmd
bash run_task.sh ${task_name} ${gpu_id}
# As example: bash run_task.sh block_hammer_beat 0
```
# 2. Task Config
Data collection configurations are located in the config folder, corresponding to each task. The most important setting is head_camera_type (default is D435), which directly affects the visual observation collected.
# 3. Decoupled Interaction Framework
Process Data for DP3 training after collecting data (In the root directory), and input the task name and the amount of data you want your policy to train with:
```cmd
python script/preprocess_pkl.py ${root_directory}
# As example: python script/preprocess_pkl.py data/block_hammer_beat_pkl/
```
```cmd
python script/pkl2zarr_dp3.py ${task_name} ${expert_data_num}
# As example: python script/pkl2zarr_dp3.py block_hammer_beat 50
```
Then, move to policy/Decoupled-Interaction-Policy first, and run the following code to train DIP:
```cmd
bash train.sh ${task_name} ${expert_data_num} ${seed} ${gpu_id}
# As example: bash train.sh block_hammer_beat 50 0 0
```
Run the following code to evaluate DIP for a specific task:
```cmd
bash eval.sh ${task_name} ${expert_data_num} ${checkpoint_num} ${seed} ${gpu_id}
# As example: bash eval.sh block_hammer_beat 50 3000 0 0
```
