deepspeed --include=localhost:5 --master_port=29555 \
    fengwu_deepspeed.py --cfg_path ./cfgs/finetune/finetune_fengwu.yaml\
    --cfg_new ./exp_script/VIT/baseline_nouncertain-lr2e-4/finetune_fengwu_new.yaml
    