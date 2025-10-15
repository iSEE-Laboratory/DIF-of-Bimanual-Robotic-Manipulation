from typing import Dict
import torch
import torch.nn as nn
from termcolor import cprint

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_policy import BasePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.model_util import print_params
from diffusion_policy.model.vision.pointnet_extractor import DP3Encoder
from diffusion_policy.model.common.fm_util import get_timesteps

class DIP(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256, 512, 1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            time_conditioning = False,
            flow_schedule = "linear",
            num_k_infer=10,
            exp_scale=4.0,
            pointcloud_encoder_cfg=None,
            **kwargs):
        super().__init__()
        
        self.condition_type = condition_type
        
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
        
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])
        
        left_obs_encoder = DP3Encoder(observation_space=obs_dict,
                                 img_crop_shape=crop_shape,
                                 out_channel=encoder_output_dim,
                                 pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                 use_pc_color=use_pc_color,
                                 pointnet_type=pointnet_type,)
        
        right_obs_encoder = DP3Encoder(observation_space=obs_dict,
                                 img_crop_shape=crop_shape,
                                 out_channel=encoder_output_dim,
                                 pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                 use_pc_color=use_pc_color,
                                 pointnet_type=pointnet_type,)
        
        obs_feature_dim = left_obs_encoder.output_shape()
        
        state_output_dim = 64
        obs_feature_dim += state_output_dim
                
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
                
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        self.time_conditioning = time_conditioning
        self.flow_schedule = flow_schedule
        self.num_k_infer = num_k_infer
        self.exp_scale = exp_scale
        
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")
        
        left_model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,)
        
        right_model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,)
        
        self.left_obs_encoder = left_obs_encoder
        self.right_obs_encoder = right_obs_encoder
        
        self.left_model = left_model
        self.right_model = right_model
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        
        self.left_scale_encoder = DP3Encoder(observation_space=obs_dict,
                                 img_crop_shape=crop_shape,
                                 out_channel=encoder_output_dim,
                                 pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                 use_pc_color=use_pc_color,
                                 pointnet_type=pointnet_type,)
        
        self.right_scale_encoder = DP3Encoder(observation_space=obs_dict,
                                 img_crop_shape=crop_shape,
                                 out_channel=encoder_output_dim,
                                 pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                 use_pc_color=use_pc_color,
                                 pointnet_type=pointnet_type,)
        
        state_input_channel = obs_feature_dim - state_output_dim
        scale_channel = 1
                
        self.left_selector = nn.Sequential(
            nn.Conv1d(state_input_channel, state_input_channel // 2, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv1d(state_input_channel // 2, scale_channel, kernel_size=1, stride=1),
            nn.Sigmoid(),)
        
        self.right_selector = nn.Sequential(
            nn.Conv1d(state_input_channel, state_input_channel // 2, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv1d(state_input_channel // 2, scale_channel, kernel_size=1, stride=1),
            nn.Sigmoid(),)

        self.left_shift = nn.Sequential(
            nn.Conv1d(state_input_channel, state_input_channel // 2, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv1d(state_input_channel // 2, state_output_dim, kernel_size=1, stride=1),)
        
        self.right_shift = nn.Sequential(
            nn.Conv1d(state_input_channel, state_input_channel // 2, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv1d(state_input_channel // 2, state_output_dim, kernel_size=1, stride=1),)
        
        self._initialize_weights()
        print_params(self)
        
    def _initialize_weights(self):
        for layer in self.left_shift:
            if isinstance(layer, nn.Conv1d):
                torch.nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu',)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        for layer in self.right_shift:
            if isinstance(layer, nn.Conv1d):
                torch.nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu',)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    def compute_loss(self, batch):
        # normalize input
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        
        nactions = nactions.to(self.device)
        nobs = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in nobs.items()}
                
        if not self.use_pc_color:
            nobs['left_point_cloud'] = nobs['left_point_cloud'][..., :3]
            nobs['right_point_cloud'] = nobs['right_point_cloud'][..., :3]
        
        batch_size = nactions.shape[0]
        
        left_global_cond = None
        left_trajectory = nactions[..., :7]
        
        right_global_cond = None
        right_trajectory = nactions[..., 7:]
            
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, 
                lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            left_nobs = {'point_cloud': this_nobs['left_point_cloud'],
                         'agent_pos': this_nobs['agent_pos'][..., :7]}
            right_nobs = {'point_cloud': this_nobs['right_point_cloud'],
                          'agent_pos': this_nobs['agent_pos'][..., 7:]}
            
            left_nobs_features = self.left_obs_encoder(left_nobs)
            right_nobs_features = self.right_obs_encoder(right_nobs)
            
            pcd_output_dim = 128
            left_add_state_features = right_nobs_features.clone().detach()[..., pcd_output_dim:]
            right_add_state_features = left_nobs_features.clone().detach()[..., pcd_output_dim:]
                        
            left_scale_features = self.left_scale_encoder(left_nobs).reshape(batch_size, -1, self.n_obs_steps)
            right_scale_features = self.right_scale_encoder(right_nobs).reshape(batch_size, -1, self.n_obs_steps)
            
            left_scale = self.left_selector(left_scale_features).reshape(batch_size * self.n_obs_steps, -1)
            right_scale = self.right_selector(right_scale_features).reshape(batch_size * self.n_obs_steps, -1)
                         
            left_add_state_features = left_scale * left_add_state_features
            right_add_state_features = right_scale * right_add_state_features
            
            left_add_state_features += self.left_shift(left_scale_features).reshape(batch_size * self.n_obs_steps, -1)
            right_add_state_features += self.right_shift(right_scale_features).reshape(batch_size * self.n_obs_steps, -1)
            
            left_nobs_features = torch.cat([left_nobs_features, left_add_state_features], dim=-1)
            right_nobs_features = torch.cat([right_nobs_features, right_add_state_features], dim=-1)
                                                       
            if "cross_attention" in self.condition_type:
                left_global_cond = left_nobs_features.reshape(batch_size, self.n_obs_steps, -1)
                right_global_cond = right_nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                left_global_cond = left_nobs_features.reshape(batch_size, -1)
                right_global_cond = right_nobs_features.reshape(batch_size, -1)
        
        left_noise = torch.randn(left_trajectory.shape, device=left_trajectory.device)
        left_t = torch.rand((batch_size, 1, 1), device=left_trajectory.device)
        left_trajectory_t = left_t * left_trajectory + (1.0 - left_t) * left_noise
        left_target_vel = left_trajectory - left_noise
        
        pos_emb_scale = 20
        left_timesteps = left_t.squeeze()
        if self.time_conditioning:
            left_timesteps = left_t.squeeze() * pos_emb_scale
        
        left_pred_vel = self.left_model(sample=left_trajectory_t,
                              timestep=left_timesteps,
                              global_cond=left_global_cond,)
        
        right_noise = torch.randn(right_trajectory.shape, device=right_trajectory.device)
        right_t = torch.rand((batch_size, 1, 1), device=right_trajectory.device)
        right_trajectory_t = right_t * right_trajectory + (1.0 - right_t) * right_noise
        right_target_vel = right_trajectory - right_noise
        
        pos_emb_scale = 20
        right_timesteps = right_t.squeeze()
        if self.time_conditioning:
            right_timesteps = right_t.squeeze() * pos_emb_scale
        
        right_pred_vel = self.right_model(sample=right_trajectory_t,
                              timestep=right_timesteps,
                              global_cond=right_global_cond,)
                
        loss = None
        loss_fun = nn.MSELoss()
        left_loss = loss_fun(left_pred_vel, left_target_vel)
        right_loss = loss_fun(right_pred_vel, right_target_vel)
        loss = left_loss + right_loss
        
        loss_dict = {'bc_loss': loss.item(),}
        
        return loss, loss_dict
    
    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        nobs = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in nobs.items()}
                
        if not self.use_pc_color:
            nobs['left_point_cloud'] = nobs['left_point_cloud'][..., :3]
            nobs['right_point_cloud'] = nobs['right_point_cloud'][..., :3]
                
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps
        
        device = self.device
        dtype = self.dtype
        
        left_global_cond = None
        right_global_cond = None
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            left_nobs = {'point_cloud': this_nobs['left_point_cloud'],
                         'agent_pos': this_nobs['agent_pos'][..., :7]}
            right_nobs = {'point_cloud': this_nobs['right_point_cloud'],
                          'agent_pos': this_nobs['agent_pos'][..., 7:]}
            
            left_nobs_features = self.left_obs_encoder(left_nobs)
            right_nobs_features = self.right_obs_encoder(right_nobs)
            
            pcd_output_dim = 128
            left_add_state_features = right_nobs_features.clone().detach()[..., pcd_output_dim:]
            right_add_state_features = left_nobs_features.clone().detach()[..., pcd_output_dim:]
                        
            left_scale_features = self.left_scale_encoder(left_nobs).reshape(B, -1, self.n_obs_steps)
            right_scale_features = self.right_scale_encoder(right_nobs).reshape(B, -1, self.n_obs_steps)
            
            left_scale = self.left_selector(left_scale_features).reshape(B * self.n_obs_steps, -1)
            right_scale = self.right_selector(right_scale_features).reshape(B * self.n_obs_steps, -1)
            
            left_add_state_features = left_scale * left_add_state_features
            right_add_state_features = right_scale * right_add_state_features

            left_add_state_features += self.left_shift(left_scale_features).reshape(B * self.n_obs_steps, -1)
            right_add_state_features += self.right_shift(right_scale_features).reshape(B * self.n_obs_steps, -1)

            left_nobs_features = torch.cat([left_nobs_features, left_add_state_features], dim=-1)
            right_nobs_features = torch.cat([right_nobs_features, right_add_state_features], dim=-1)
                           
            if "cross_attention" in self.condition_type:
                left_global_cond = left_nobs_features.reshape(B, self.n_obs_steps, -1)
                right_global_cond = right_nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                left_global_cond = left_nobs_features.reshape(B, -1)
                right_global_cond = right_nobs_features.reshape(B, -1)
        
            left_cond_data = torch.randn(size=(B, T, Da), device=device, dtype=dtype)
            right_cond_data = torch.randn(size=(B, T, Da), device=device, dtype=dtype)
        
        left_traj = [left_cond_data]
        left_t0, left_dt = get_timesteps(self.flow_schedule, 
                               self.num_k_infer, 
                               exp_scale=self.exp_scale,)
        
        right_traj = [right_cond_data]
        right_t0, right_dt = get_timesteps(self.flow_schedule, 
                               self.num_k_infer, 
                               exp_scale=self.exp_scale,)
        
        
        for i in range(self.num_k_infer):
            left_timesteps = torch.ones((B), device=device) * left_t0[i]
            if self.time_conditioning:
                pos_emb_scale = 20
                left_timesteps *= pos_emb_scale
            
            left_vel_pred = self.left_model(sample=left_cond_data,
                                  timestep=left_timesteps,
                                  global_cond=left_global_cond,)
            
            left_cond_data = left_cond_data.detach().clone() + left_vel_pred * left_dt[i]
            
            right_timesteps = torch.ones((B), device=device) * right_t0[i]
            if self.time_conditioning:
                pos_emb_scale = 20
                right_timesteps *= pos_emb_scale
            
            right_vel_pred = self.right_model(sample=right_cond_data,
                                  timestep=right_timesteps,
                                  global_cond=right_global_cond,)
            
            right_cond_data = right_cond_data.detach().clone() + right_vel_pred * right_dt[i]
                     
        left_traj.append(left_cond_data)
        right_traj.append(right_cond_data)
                
        # unnormalize prediction
        traj = torch.cat([left_traj[-1], right_traj[-1]], dim=-1)
        naction_pred = traj
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        
        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred,}
        
        return result
            