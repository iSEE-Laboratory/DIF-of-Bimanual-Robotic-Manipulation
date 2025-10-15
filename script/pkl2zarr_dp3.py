import pdb, pickle, os
import numpy as np
import open3d as o3d
from copy import deepcopy
import zarr, shutil
import argparse

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('task_name', type=str)
    parser.add_argument('episode_number', type=int)

    args = parser.parse_args()
    
    visualize_pcd = False

    task_name = args.task_name
    num = args.episode_number
    current_ep, num = 0, num
    load_dir = f'./data/{task_name}_pkl'
    
    total_count = 0

    save_dir = f'./policy/Decoupled-Interaction-Policy/Decoupled-Interaction-Policy/data/{task_name}_{num}.zarr'
    
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    
    episode_ends_arrays, action_arrays, state_arrays, joint_action_arrays = [], [], [], []
    left_point_cloud_arrays = []
    right_point_cloud_arrays = []
    
    while os.path.isdir(load_dir+f'/episode{current_ep}') and current_ep < num:
        print(f'processing episode: {current_ep + 1} / {num}', end='\r')
        file_num = 0
        state_sub_arrays = []
        action_sub_arrays = [] 
        joint_action_sub_arrays = []
        episode_ends_sub_arrays = []
        
        left_point_cloud_sub_arrays = []
        right_point_cloud_sub_arrays = []
        
        while os.path.exists(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl'):
            with open(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl', 'rb') as file:
                data = pickle.load(file)
                
            action = data['endpose']
            joint_action = data['joint_action']

            state_sub_arrays.append(joint_action)
            action_sub_arrays.append(action)
            joint_action_sub_arrays.append(joint_action)
            
            left_pcd = data['left_pointcloud'][:, :]
            right_pcd = data['right_pointcloud'][:, :]
            
            left_point_cloud_sub_arrays.append(left_pcd)
            right_point_cloud_sub_arrays.append(right_pcd)
            
            if visualize_pcd:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(data['pcd']['points'])
                pcd.colors = o3d.utility.Vector3dVector(data['pcd']['colors'])
                o3d.visualization.draw_geometries([pcd])

            file_num += 1
            total_count += 1
            
        current_ep += 1
        
        episode_ends_arrays.append(deepcopy(total_count))
        action_arrays.extend(action_sub_arrays)
        state_arrays.extend(state_sub_arrays)
        joint_action_arrays.extend(joint_action_sub_arrays)
        
        left_point_cloud_arrays.extend(left_point_cloud_sub_arrays)
        right_point_cloud_arrays.extend(right_point_cloud_sub_arrays)
        
    print()
    episode_ends_arrays = np.array(episode_ends_arrays)
    action_arrays = np.array(action_arrays)
    state_arrays = np.array(state_arrays)
    joint_action_arrays = np.array(joint_action_arrays)
    
    left_point_cloud_arrays = np.array(left_point_cloud_arrays)
    right_point_cloud_arrays = np.array(right_point_cloud_arrays)
    
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    action_chunk_size = (100, action_arrays.shape[1])
    state_chunk_size = (100, state_arrays.shape[1])
    joint_chunk_size = (100, joint_action_arrays.shape[1])
    
    left_point_cloud_chunk_size = (100, left_point_cloud_arrays.shape[1], left_point_cloud_arrays.shape[2])
    right_point_cloud_chunk_size = (100, right_point_cloud_arrays.shape[1], right_point_cloud_arrays.shape[2])
    
    zarr_data.create_dataset('tcp_action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=joint_action_arrays, chunks=joint_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
    
    zarr_data.create_dataset('left_point_cloud', data=left_point_cloud_arrays, chunks=left_point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('right_point_cloud', data=right_point_cloud_arrays, chunks=right_point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)

if __name__ == '__main__':
    main()
