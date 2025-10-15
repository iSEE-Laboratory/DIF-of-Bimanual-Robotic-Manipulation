# use the same command as training except the script
# for example:
# bash script/run_eval_policy_dp3.sh robot_dp3 $task_name $expert_data_num eval $seed $checkpoint_num $gpu_id

DEBUG=False

alg_name=${1}
task_name=${2}
expert_data_num=${3}
config_name=${alg_name}
addition_info=${4}
seed=${5}
checkpoint_num=${6}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="./policy/Decoupled-Interaction-Policy/Decoupled-Interaction-Policy/diffusion_policy/data/outputs/${exp_name}_seed${seed}"

gpu_id=${7}

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
python script/eval_policy_dp3.py --config-name=${config_name}.yaml \
                            task=${task_name}_${expert_data_num} \
                            raw_task_name=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint_num=${checkpoint_num} \
                            expert_data_num=${expert_data_num}
