deepspeed --include=localhost:7 --master_port=29577 \
    fengwu_deepspeed.py --cfg_path ./cfgs/finetune/finetune_fengwu.yaml\
    --cfg_new ./exp_script/baseline_useyl/finetune_fengwu_new.yaml