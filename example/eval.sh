
cfg_path=/home/gaoziyi/weather/weather_forcast/cfgs/multivariable_deepspeed.yaml

ckpt_path=/home/gaoziyi/weather/deepspeed_checkpoint/swin-run-384-patch1/step_350000/mp_rank_00_model_states.pt
device=$1

echo $cfg_path
echo $ckpt_path

python eval.py --cfg_path $cfg_path \
 --ckpt_path $ckpt_path \
 --device $device

last_three_folders=$(echo $ckpt_path | awk -F/ '{n=NF; print $(n-2)"_"$(n-1)}')

echo $last_three_folders

python visualize.py --path $last_three_folders