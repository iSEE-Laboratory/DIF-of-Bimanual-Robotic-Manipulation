
from .base_task import Base_task
from .utils import *
import sapien
import math

class blocks_stack_easy(Base_task):

    def setup_demo(self,**kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera(kwags.get('camera_w', 640),kwags.get('camera_h', 480))
        self.pre_move()
        self.load_actors()
        self.step_lim = 1000

    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def load_actors(self):
        block_pose = rand_pose(
            xlim=[-0.25,0.25],
            ylim=[-0.15,0.05],
            zlim=[0.76],
            qpos=[0.5, 0.5, 0.5, 0.5],
            ylim_prop=True,
            rotate_rand=True,
            rotate_lim=[0,1.57,0],
        )

        while abs(block_pose.p[0]) < 0.05 or np.sum(pow(block_pose.p[:2] - np.array([0,-0.1]),2)) < 0.0225:
            block_pose = rand_pose(
                xlim=[-0.25,0.25],
                ylim=[-0.15,0.05],
                zlim=[0.76],
                qpos=[0.5, 0.5, 0.5, 0.5],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0,1.57,0],
            )

        self.block1 = create_box(
            scene = self.scene,
            pose = block_pose,
            half_size=(0.025,0.025,0.025),
            color=(1,0,0),
            name="box"
        )

        block_pose = rand_pose(
            xlim=[-0.25,0.25],
            ylim=[-0.15,0.05],
            zlim=[0.76],
            qpos=[0.5, 0.5, 0.5, 0.5],
            ylim_prop=True,
            rotate_rand=True,
            rotate_lim=[0,1.57,0],
        )

        while abs(block_pose.p[0]) < 0.05 or np.sum(pow(block_pose.p[:2] - self.block1.get_pose().p[:2],2)) < 0.01 \
              or np.sum(pow(block_pose.p[:2] - np.array([0,-0.1]),2)) < 0.0225:
            block_pose = rand_pose(
                xlim=[-0.25,0.25],
                ylim=[-0.15,0.05],
                zlim=[0.76],
                qpos=[0.5, 0.5, 0.5, 0.5],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0,1.57,0],
            )


        self.block2 = create_box(
            scene = self.scene,
            pose = block_pose,
            half_size=(0.025,0.025,0.025),
            color=(0,0,0),
            name="box"
        )

        self.block1.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01
        self.block2.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01

    def play_once(self):
        pass
        
    def check_success(self):
        block1_pose = self.block1.get_pose().p
        block2_pose = self.block2.get_pose().p
        target_pose = [0,-0.1]
        eps = [0.025,0.025,0.01]
        # return 1
        return np.all(abs(block1_pose - np.array(target_pose + [0.765])) < eps) and \
               np.all(abs(block2_pose - np.array(target_pose + [0.815])) < eps) and self.is_left_gripper_open() and self.is_right_gripper_open()