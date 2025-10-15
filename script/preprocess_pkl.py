import sys
sys.path.append('./')
sys.path.append('./envs')

import os
import pickle
import argparse

import numpy as np
from envs.utils import fps

def get_pointclouds(data):
    bound = [[-0.6, -0.35, 0.745],[0.6, 0.35, 2]]
    min_bound = bound[0]
    max_bound = bound[1]
    
    head_intrinsic = data['observation']['head_camera']['intrinsic_cv']
    h, w = data['observation']['head_camera']['depth'].shape
    fx, fy = head_intrinsic[0, 0], head_intrinsic[1, 1]
    cx, cy = head_intrinsic[0, 2], head_intrinsic[1, 2]
    
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = -data['observation']['head_camera']['depth'] / 1000.
    x = (u - cx) * z / fx
    x = -x
    y = (v - cy) * z / fy
    head_points_camera = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    model_matrix = data['observation']['head_camera']['cam2world_gl']
    head_points_world = head_points_camera @ model_matrix[:3, :3].T + model_matrix[:3, 3]
    inside_bounds_mask = (head_points_world > min_bound).all(axis=1) & (head_points_world < max_bound).all(axis=1)
    head_points_world_cropped = head_points_world[inside_bounds_mask]
    head_points_world_cropped, index = fps(head_points_world_cropped, 1024)
    index = index.cpu().numpy()[0]
    
    head_points_world_cropped_rgb = data['observation']['head_camera']['rgb'] / 255.0
    head_points_world_cropped_rgb = head_points_world_cropped_rgb.reshape(-1, 3)[inside_bounds_mask][index, :]
    head_pointcloud = np.hstack((head_points_world_cropped, head_points_world_cropped_rgb))
    
    left_intrinsic = data['observation']['left_camera']['intrinsic_cv']
    h, w = data['observation']['left_camera']['depth'].shape
    fx, fy = left_intrinsic[0, 0], left_intrinsic[1, 1]
    cx, cy = left_intrinsic[0, 2], left_intrinsic[1, 2]
    
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = -data['observation']['left_camera']['depth'] / 1000.
    x = (u - cx) * z / fx
    x = -x
    y = (v - cy) * z / fy
    left_points_camera = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    model_matrix = data['observation']['left_camera']['cam2world_gl']
    left_points_world = left_points_camera @ model_matrix[:3, :3].T + model_matrix[:3, 3]
    inside_bounds_mask = (left_points_world > min_bound).all(axis=1) & (left_points_world < max_bound).all(axis=1)
    left_points_world_cropped = left_points_world[inside_bounds_mask]
    left_points_world_cropped, index = fps(left_points_world_cropped, 1024)
    index = index.cpu().numpy()[0]
    
    left_points_world_cropped_rgb = data['observation']['left_camera']['rgb'] / 255.0
    left_points_world_cropped_rgb = left_points_world_cropped_rgb.reshape(-1, 3)[inside_bounds_mask][index, :]
    left_points = np.hstack((left_points_world_cropped, left_points_world_cropped_rgb))
    
    right_intrinsic = data['observation']['right_camera']['intrinsic_cv']
    h, w = data['observation']['right_camera']['depth'].shape
    fx, fy = right_intrinsic[0, 0], right_intrinsic[1, 1]
    cx, cy = right_intrinsic[0, 2], right_intrinsic[1, 2]
    
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = -data['observation']['right_camera']['depth'] / 1000.
    x = (u - cx) * z / fx
    x = -x
    y = (v - cy) * z / fy
    right_points_camera = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    model_matrix = data['observation']['right_camera']['cam2world_gl']
    right_points_world = right_points_camera @ model_matrix[:3, :3].T + model_matrix[:3, 3]
    inside_bounds_mask = (right_points_world > min_bound).all(axis=1) & (right_points_world < max_bound).all(axis=1)
    right_points_world_cropped = right_points_world[inside_bounds_mask]
    right_points_world_cropped, index = fps(right_points_world_cropped, 1024)
    index = index.cpu().numpy()[0]
    
    right_points_world_cropped_rgb = data['observation']['right_camera']['rgb'] / 255.0
    right_points_world_cropped_rgb = right_points_world_cropped_rgb.reshape(-1, 3)[inside_bounds_mask][index, :]
    right_points = np.hstack((right_points_world_cropped, right_points_world_cropped_rgb))
    
    left_points = np.concatenate([head_pointcloud, left_points], axis=0)
    right_points = np.concatenate([head_pointcloud, right_points], axis=0)
    
    return {
        'left_pointcloud': left_points,
        'right_pointcloud': right_points,
    }
    
def add_elements_to_pkl_in_directory(root_directory):
    for subdir, _, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".pkl"):
                file_path = os.path.join(subdir, file)
                
                with open(file_path, "rb") as pkl_file:
                    data = pickle.load(pkl_file)
                
                data_update = get_pointclouds(data)
                data.update(data_update)
                
                with open(file_path, "wb") as pkl_file:
                    pickle.dump(data, pkl_file)
                
                print(f"Updated: {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_directory", type=str)
    args = parser.parse_args()
    add_elements_to_pkl_in_directory(root_directory=args.root_directory)
    