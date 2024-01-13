
cfg_path=/home/gaoziyi/weather/weather_forcast/cfgs/multivariable_deepspeed.yaml

ckpt_path=$2
device=$1

echo $cfg_path
echo $ckpt_path

python valid_onstep.py --cfg_path $cfg_path \
 --ckpt_path $ckpt_path \
 --device $device

last_three_folders=$(echo $ckpt_path | awk -F/ '{n=NF; print $(n-2)"_"$(n-1)}')

echo $last_three_folders

python visualize.py --path $last_three_folders