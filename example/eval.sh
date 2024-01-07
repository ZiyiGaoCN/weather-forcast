
cfg_path=/home/gaoziyi/weather/weather_forcast/cfgs/multivariable_deepspeed.yaml

ckpt_path=/home/gaoziyi/weather/deepspeed_checkpoint/AFNO-uncertain/step_38000/mp_rank_00_model_states.pt
ckpt_path=/home/gaoziyi/weather/deepspeed_checkpoint/AFNO-gcloss/step_18000/mp_rank_00_model_states.pt
ckpt_path=/home/gaoziyi/weather/deepspeed_checkpoint/SFNO-gcloss/step_32000/mp_rank_00_model_states.pt
ckpt_path=/home/gaoziyi/weather/deepspeed_checkpoint/singlevariable-1024/step_62000/mp_rank_00_model_states.pt
ckpt_path=/home/gaoziyi/weather/deepspeed_checkpoint/singlevariable-1024-nouncertain/step_72000/mp_rank_00_model_states.pt
ckpt_path=/home/gaoziyi/weather/deepspeed_checkpoint/singlevariable-1024/step_72000/mp_rank_00_model_states.pt

device=$1

echo $cfg_path
echo $ckpt_path

python eval.py --cfg_path $cfg_path \
 --ckpt_path $ckpt_path \
 --device $device

last_three_folders=$(echo $ckpt_path | awk -F/ '{n=NF; print $(n-2)"_"$(n-1)}')

echo $last_three_folders

python visualize.py --path $last_three_folders