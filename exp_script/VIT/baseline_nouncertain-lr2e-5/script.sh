deepspeed --include=localhost:4 --master_port=29544 \
    fengwu_deepspeed.py --cfg_path ./cfgs/finetune/finetune_fengwu.yaml\
    --cfg_new ./exp_script/VIT/baseline_nouncertain-lr2e-5/finetune_fengwu_new.yaml
    