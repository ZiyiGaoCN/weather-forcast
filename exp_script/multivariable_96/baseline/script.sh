deepspeed --include=localhost:6 --master_port=29566 \
    fengwu_deepspeed.py --cfg_path ./cfgs/finetune/finetune_fengwu.yaml\
    --cfg_new ./exp_script/baseline/finetune_fengwu_new.yaml